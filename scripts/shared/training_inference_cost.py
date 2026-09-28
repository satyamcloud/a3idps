# Training/inference cost comparison: A3IDPS's full pipeline (teacher -> distilled baseline -> 5 feedback
# rounds, each round = fresh AdvGAN generator training + classifier retraining) vs. standard PGD adversarial
# training (one training loop with PGD examples generated on the fly), on both datasets.
#
# This is a COST characterization, not a robustness result - one fresh seed-0 run per dataset, timed stage by
# stage with wall-clock time.time(). It does NOT overwrite the production checkpoints in models/{nsl_v2,
# cicids_v2}/: fresh models trained here are thrown away after timing (not saved), so this script is safe to
# run without touching anything the paper's actual numbers depend on.
#
# Uses the exact same architecture, batch size, epochs/patience, generator steps, and the SAME eps values
# already selected on validation for the real pipeline (read from the existing stage2b_selected /
# stage3_selection JSONs) - so the comparison is apples-to-apples at the operating point actually used in the
# paper, not some other hyperparameter setting.
#
# Inference latency is measured separately and needs NO training: it just times batched forward passes on
# the already-saved seed-0 models (undefended, fgsm_at, pgd_at, kd_feedback_r5) over the full test set.
#
# Expect this to take a while for the A3IDPS side (teacher + baseline + 5 x (generator + classifier), each
# with early stopping) - that IS the point being measured. Run one dataset at a time if Colab session length
# is a concern; each dataset's timing is saved to Drive as soon as it finishes.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, json, random, time
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
SEED = 0
GEN_STEPS, LAM_CLS, LAM_L1, LAM_GAN, KAPPA = 1200, 20.0, 0.5, 1.0, 1.0
ROUNDS = 5
KD_ALPHA, KD_T = 0.7, 3.0

DATASETS = {
    "nsl": {
        "data": f"{BASE}/data_processed/nslkdd_v2/nsl_v2_arrays.npz",
        "mod": f"{BASE}/models/nsl_v2", "res": f"{BASE}/results/nsl_v2",
        "batch": 512, "max_epochs": 40, "feedback_epochs": 30,
        "base_lr": 1e-3, "feedback_lr": 1e-3, "class_weight_cap": 20.0,
        "categorical_mask": True,
    },
    "cicids": {
        "data": f"{BASE}/data_processed/cicids_v2/cicids_v2_primary.npz",
        "mod": f"{BASE}/models/cicids_v2", "res": f"{BASE}/results/cicids_v2",
        "batch": 2048, "max_epochs": 60, "feedback_epochs": 30,
        "base_lr": 3e-4, "feedback_lr": 1.5e-4, "class_weight_cap": 20.0,
        "categorical_mask": False,
    },
}


def set_seed(s): random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)


def build_model(d, ncls, widths=(256, 128, 64)):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(widths[0])(inp); x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dense(widths[1])(x);   x = layers.Dropout(0.3)(x);          x = layers.Activation("relu")(x)
    x = layers.Dense(widths[2])(x);   x = layers.Dropout(0.3)(x);          x = layers.Activation("relu")(x)
    return keras.Model(inp, layers.Dense(ncls)(x))


def build_generator(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256, activation="relu")(inp); x = layers.Dense(256, activation="relu")(x)
    return keras.Model(inp, layers.Dense(d)(x))


def build_disc(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256)(inp); x = layers.LeakyReLU(0.2)(x); x = layers.Dense(256)(x); x = layers.LeakyReLU(0.2)(x)
    return keras.Model(inp, layers.Dense(1)(x))


def make_fgsm(model, mask):
    M = tf.constant(mask, tf.float32)
    def loss_fn(z, y):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)
    @tf.function
    def fgsm(x0, y, eps):
        with tf.GradientTape() as t:
            t.watch(x0); s = tf.reduce_sum(loss_fn(model(x0, training=False), y))
        return tf.clip_by_value(x0 + eps * M * tf.sign(t.gradient(s, x0)), 0., 1.)
    return fgsm


def make_pgd(model, mask, ncls):
    M = tf.constant(mask, tf.float32)
    def loss_fn(z, y):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)
    @tf.function
    def pgd(x0, y, eps, steps=10, restarts=1):
        E = eps * M; best_x, best_l = x0, tf.fill(tf.shape(y), -1e30)
        for r in range(restarts):
            x = x0 if r == 0 else tf.clip_by_value(x0 + tf.random.uniform(tf.shape(x0), -1., 1.) * E, 0., 1.)
            for s_ in range(steps):
                alpha = (1.0 if s_ < steps // 2 else 0.25) * E / 4.0
                with tf.GradientTape() as t:
                    t.watch(x); s = tf.reduce_sum(loss_fn(model(x, training=False), y))
                x = tf.clip_by_value(tf.clip_by_value(x + alpha * tf.sign(t.gradient(s, x)), x0 - E, x0 + E), 0., 1.)
                l_new = loss_fn(model(x, training=False), y)
                upd = l_new > best_l
                best_x, best_l = tf.where(upd[:, None], x, best_x), tf.where(upd, l_new, best_l)
        return best_x
    return pgd


def fit_classifier(model, X, y, Xv, yv, cw, batch, lr, max_epochs, patience=8, seed=SEED,
                    X_extra=None, Xv_extra=None, zt=None, at_pgd=None, at_fgsm=None, at_eps=None, tag=""):
    set_seed(seed)
    opt = keras.optimizers.Adam(lr); opt.build(model.trainable_variables)
    use_kd, cwt = zt is not None, tf.constant(cw)
    et = tf.constant(at_eps, tf.float32) if at_eps is not None else None

    def wce(z, yy):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=z) * tf.gather(cwt, yy))

    @tf.function(reduce_retracing=True)
    def step(xb, yb, ztb):
        with tf.GradientTape() as t:
            z = model(xb, training=True)
            hard = wce(z, yb)
            if use_kd:
                pt = tf.nn.softmax(ztb / KD_T); log_ps = tf.nn.log_softmax(z / KD_T)
                kl = tf.reduce_mean(tf.reduce_sum(pt * (tf.math.log(pt + 1e-8) - log_ps), axis=1))
                loss = KD_ALPHA * hard + (1. - KD_ALPHA) * KD_T * KD_T * kl
            else:
                loss = hard
        opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    rng, n = np.random.RandomState(seed), len(X)
    best, best_w, wait, cur_lr, epochs_run = 1e9, None, 0, lr, 0
    for ep in range(max_epochs):
        perm = rng.permutation(n)
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i+batch]; xb, yb = X[idx], y[idx]
            if X_extra is not None:
                xb, yb = np.concatenate([xb, X_extra[idx]]), np.concatenate([yb, y[idx]])
            if at_pgd is not None:
                xa = at_pgd(tf.constant(X[idx]), tf.constant(y[idx]), eps=et)
                xb, yb = np.concatenate([xb, xa.numpy()]), np.concatenate([yb, y[idx]])
            ztb = zt[idx] if use_kd else np.zeros((len(xb), 1), "float32")
            step(tf.constant(xb), tf.constant(yb), tf.constant(ztb))
        vl = float(wce(model(tf.constant(Xv), training=False), tf.constant(yv)))
        if at_fgsm is not None:
            xv = at_fgsm(tf.constant(Xv), tf.constant(yv), eps=et).numpy()
            vl += float(wce(model(tf.constant(xv), training=False), tf.constant(yv)))
        epochs_run = ep + 1
        if vl < best - 1e-4: best, best_w, wait = vl, model.get_weights(), 0
        else:
            wait += 1
            if wait % 4 == 0: cur_lr *= 0.5; opt.learning_rate = cur_lr
            if wait >= patience: break
    model.set_weights(best_w)
    return model, epochs_run


def train_generator(clf, X, y, eps, D, ncls, mask, batch, seed=SEED, tag=""):
    set_seed(seed)
    G, Dn = build_generator(D), build_disc(D)
    g_opt = keras.optimizers.Adam(1e-3, beta_1=0.5); g_opt.build(G.trainable_variables)
    d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5); d_opt.build(Dn.trainable_variables)
    E = tf.constant(eps * mask)
    bce = lambda labels, logits: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    def gen(x): return tf.clip_by_value(x + E * tf.tanh(G(x, training=True)), 0., 1.)

    @tf.function(reduce_retracing=True)
    def step(x, yb):
        with tf.GradientTape() as td:
            xa = gen(x)
            ld = bce(tf.ones([tf.shape(x)[0], 1]), Dn(x, training=True)) + bce(tf.zeros([tf.shape(x)[0], 1]), Dn(xa, training=True))
        d_opt.apply_gradients(zip(td.gradient(ld, Dn.trainable_variables), Dn.trainable_variables))
        with tf.GradientTape() as tg:
            xa = gen(x); z = clf(xa, training=False); oh = tf.one_hot(yb, ncls)
            margin = tf.reduce_sum(z * oh, 1) - tf.reduce_max(z - 1e9 * oh, 1)
            lg = (LAM_CLS * tf.reduce_mean(tf.nn.relu(margin + KAPPA))
                  + LAM_GAN * bce(tf.ones([tf.shape(x)[0], 1]), Dn(xa, training=False))
                  + LAM_L1 * tf.reduce_mean(tf.abs(xa - x)))
        g_opt.apply_gradients(zip(tg.gradient(lg, G.trainable_variables), G.trainable_variables))
        return lg

    rng = np.random.RandomState(seed)
    for s in range(GEN_STEPS):
        idx = rng.randint(0, len(X), batch)
        step(tf.constant(X[idx]), tf.constant(y[idx]))
    return G


def apply_G(G, X, eps, mask, bs=8192):
    E = eps * mask
    return np.vstack([np.clip(X[i:i+bs] + E * np.tanh(G(tf.constant(X[i:i+bs]), training=False).numpy()), 0., 1.)
                      for i in range(0, len(X), bs)]).astype("float32")


def time_inference(model, X, batch_size, reps=5):
    _ = model.predict(X[:batch_size], batch_size=batch_size, verbose=0)  # warm up (build the tf.function graph)
    times = []
    for _ in range(reps):
        t0 = time.time()
        model.predict(X, batch_size=batch_size, verbose=0)
        times.append(time.time() - t0)
    total_n = len(X)
    return {"seconds_per_pass": float(np.mean(times)), "std_seconds": float(np.std(times)),
            "n_samples": total_n, "ms_per_1000_samples": float(np.mean(times) / total_n * 1000 * 1000)}


# %% [1] Run for each dataset
for ds_name, cfg in DATASETS.items():
    print(f"\n========== {ds_name.upper()} ==========")
    MOD, RES = cfg["mod"], cfg["res"]
    d = np.load(cfg["data"])
    Xtr, Xva, Xte = [d[k].astype("float32") for k in ("Xtr", "Xva", "Xte")]
    ytr, yva, yte = [d[k].astype("int32") for k in ("ytr", "yva", "yte")]
    D, ncls, z_ncls = Xtr.shape[1], int(ytr.max()) + 1, int(ytr.max()) + 1

    if cfg["categorical_mask"]:
        names = json.load(open(f"{BASE}/data_processed/nslkdd_v2/feature_names.json"))
        mask = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
    else:
        mask = np.ones(D, "float32")
    cw = np.minimum(compute_class_weight("balanced", classes=np.arange(ncls), y=ytr), cfg["class_weight_cap"]).astype("float32")

    # Read the SAME eps values already selected on validation for the real pipeline.
    if ds_name == "cicids":
        eps_pgd = json.load(open(f"{RES}/stage2b_selected_seed0.json"))["pgd"]
        eps_max = json.load(open(f"{RES}/stage3_selection_seed0.json"))["selected_eps"]
    else:
        s2 = json.load(open(f"{RES}/stage2_test_seed0.json"))
        eps_pgd = s2["selected"]["pgd"]
        eps_max = json.load(open(f"{RES}/stage3_selection_seed0.json"))["selected_eps"]
    print(f"using eps_pgd={eps_pgd}, generator eps_max={eps_max} (same as production pipeline)")

    timing = {"eps_pgd": eps_pgd, "eps_max": eps_max, "stages": {}}

    # ---- PGD-AT: one training run, adversarial examples generated on the fly ----
    set_seed(SEED)
    m_pgd = build_model(D, ncls)
    pgd_atk = make_pgd(m_pgd, mask, ncls)
    fgsm_atk = make_fgsm(m_pgd, mask)
    t0 = time.time()
    m_pgd, epochs = fit_classifier(m_pgd, Xtr, ytr, Xva, yva, cw, cfg["batch"], cfg["base_lr"], cfg["max_epochs"],
                                    seed=SEED, at_pgd=pgd_atk, at_fgsm=fgsm_atk, at_eps=eps_pgd, tag="pgd_at")
    t_pgd = time.time() - t0
    timing["stages"]["pgd_at_training"] = {"seconds": round(t_pgd, 1), "epochs_run": epochs}
    print(f"PGD-AT training: {t_pgd:.1f}s ({epochs} epochs)")

    # ---- A3IDPS: teacher -> distilled baseline -> 5 feedback rounds ----
    t0 = time.time()
    teacher = build_model(D, ncls, widths=(512, 256, 128))
    teacher, t_ep = fit_classifier(teacher, Xtr, ytr, Xva, yva, cw, cfg["batch"], cfg["base_lr"], cfg["max_epochs"],
                                    seed=SEED, tag="teacher")
    t_teacher = time.time() - t0
    timing["stages"]["a3idps_teacher"] = {"seconds": round(t_teacher, 1), "epochs_run": t_ep}
    print(f"A3IDPS teacher training: {t_teacher:.1f}s ({t_ep} epochs)")

    zt_tr = teacher.predict(Xtr, batch_size=8192, verbose=0).astype("float32")
    t0 = time.time()
    student = build_model(D, ncls)
    student, s_ep = fit_classifier(student, Xtr, ytr, Xva, yva, cw, cfg["batch"], cfg["base_lr"], cfg["max_epochs"],
                                    seed=SEED, zt=zt_tr, tag="distilled_baseline")
    t_distill = time.time() - t0
    timing["stages"]["a3idps_distillation"] = {"seconds": round(t_distill, 1), "epochs_run": s_ep}
    print(f"A3IDPS distillation (baseline): {t_distill:.1f}s ({s_ep} epochs)")

    prev = student
    round_times = []
    for r in range(1, ROUNDS + 1):
        t0 = time.time()
        G = train_generator(prev, Xtr, ytr, eps_max, D, ncls, mask, cfg["batch"], seed=SEED * 100 + r, tag=f"gen_r{r}")
        t_gen = time.time() - t0
        Xa_tr, Xa_va = apply_G(G, Xtr, eps_max, mask), apply_G(G, Xva, eps_max, mask)
        t0 = time.time()
        new = build_model(D, ncls); new.set_weights(prev.get_weights())
        new, r_ep = fit_classifier(new, Xtr, ytr, Xva, yva, cw, cfg["batch"], cfg["feedback_lr"], cfg["feedback_epochs"],
                                    seed=SEED * 100 + r, X_extra=Xa_tr, Xv_extra=Xa_va, tag=f"round_r{r}")
        t_cls = time.time() - t0
        round_times.append({"round": r, "generator_seconds": round(t_gen, 1),
                             "classifier_seconds": round(t_cls, 1), "classifier_epochs_run": r_ep})
        print(f"  round {r}: generator={t_gen:.1f}s, classifier retrain={t_cls:.1f}s ({r_ep} epochs)")
        prev = new
    timing["stages"]["a3idps_feedback_rounds"] = round_times

    a3idps_total = t_teacher + t_distill + sum(rt["generator_seconds"] + rt["classifier_seconds"] for rt in round_times)
    timing["totals"] = {"pgd_at_seconds": round(t_pgd, 1), "a3idps_full_pipeline_seconds": round(a3idps_total, 1),
                         "a3idps_to_pgd_at_ratio": round(a3idps_total / t_pgd, 2) if t_pgd > 0 else None}
    print(f"TOTAL: PGD-AT={t_pgd:.1f}s | A3IDPS full pipeline={a3idps_total:.1f}s "
          f"({timing['totals']['a3idps_to_pgd_at_ratio']}x)")

    # ---- Inference latency: EXISTING saved seed-0 models, no training needed ----
    if ds_name == "cicids":
        eps_fgsm = json.load(open(f"{RES}/stage2b_selected_seed0.json"))["fgsm"]
        infer_paths = {
            "undefended": f"{MOD}/undefended_smote_seed0.keras",
            "fgsm_at": f"{MOD}/fgsm_eps{eps_fgsm}_seed0.keras",
            "pgd_at": f"{MOD}/pgd_eps{eps_pgd}_seed0.keras",
            "kd_feedback_r5": f"{MOD}/s3_kd_feedback_eps{eps_max}_r5_seed0.keras",
        }
    else:
        eps_fgsm = json.load(open(f"{RES}/stage2_test_seed0.json"))["selected"]["fgsm"]
        infer_paths = {
            "undefended": f"{MOD}/s2_none_seed0.keras",
            "fgsm_at": f"{MOD}/s2_fgsm_eps{eps_fgsm}_seed0.keras",
            "pgd_at": f"{MOD}/s2_pgd_eps{eps_pgd}_seed0.keras",
            "kd_feedback_r5": f"{MOD}/s4_kd_feedback_r5_seed0.keras",
        }
    inference = {}
    for tag, path in infer_paths.items():
        mdl = keras.models.load_model(path, compile=False)
        inference[tag] = time_inference(mdl, Xte, batch_size=8192)
        print(f"  inference [{tag}]: {inference[tag]['ms_per_1000_samples']:.3f} ms/1000 samples "
              f"(n={inference[tag]['n_samples']})")
    timing["inference"] = inference

    out_path = f"{RES}/training_inference_cost.json"
    json.dump(timing, open(out_path, "w"), indent=1)
    print(f"saved {out_path}")

print("\nDone. Fresh models trained here were NOT saved to models/ - only timing was recorded; "
      "production checkpoints are untouched.")
