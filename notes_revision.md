# Notes de révision — Neural Network Project

## Phase 1 — Activations

### 1.1 ReLU (Rectified Linear Unit)

**Définition mathématique**

ReLU(z) = max(0, z) = { z  si z > 0
                     { 0  sinon

**Dérivée**

ReLU'(z) = { 1  si z > 0
           { 0  si z ≤ 0

En z = 0 la dérivée n'existe pas formellement (point anguleux). Par convention
on prend 0. Probabilité d'avoir exactement z = 0 en float : nulle, donc OK.

**Pourquoi ReLU plutôt qu'une autre activation ?**

1. Non-linéaire — condition nécessaire pour que le réseau apprenne des
   frontières de décision non-linéaires. Sans non-linéarité, l'empilement
   de couches collapse :
        z₂ = (X W₁ + b₁) W₂ + b₂ = X (W₁W₂) + (b₁W₂ + b₂) = X W' + b'
   → équivalent à une simple régression linéaire.

2. Calcul trivial — juste un max, pas d'exponentielle. Beaucoup plus rapide
   que sigmoid ou tanh sur de grands réseaux.

3. Évite le vanishing gradient — pour z > 0, ReLU'(z) = 1 exactement.
   Le gradient se propage intact à travers les couches.
   Comparaison : sigmoid'(z) max = 0.25, donc à chaque couche le gradient
   est au mieux divisé par 4. Sur 10 couches : facteur 4^10 ≈ 10⁻⁶.
   → ReLU résout ce problème.

**Limite connue : "dying ReLU"**

Si trop de neurones se retrouvent avec z ≤ 0 en permanence, leur dérivée
vaut 0 et ils n'apprennent plus jamais (gradient bloqué). C'est le revers
du "gradient = 0 si z ≤ 0".

**Choix d'implémentation**

- `np.maximum(0, z)` : opération vectorisée, comparaison élément par élément.
  À ne pas confondre avec `np.max(z)` qui réduit à un scalaire.
- Pour la dérivée, `(z > 0).astype(float)` exploite le fait qu'un booléen
  numpy converti en float donne 0.0 ou 1.0 — exactement la dérivée.
- Aucune boucle Python : tout est vectorisé, conforme à l'exigence du
  brief (point "Advanced requirements").

### 1.2 Tanh (tangente hyperbolique)

**Définition mathématique**

tanh(z) = (e^z - e^-z) / (e^z + e^-z)

Propriétés :
- Bornée : tanh(z) ∈ ]-1, 1[
- Impaire : tanh(-z) = -tanh(z)
- Centrée en 0 : tanh(0) = 0
- Asymptotes : tanh(z) → 1 quand z → +∞, tanh(z) → -1 quand z → -∞

**Dérivée**

tanh'(z) = 1 - tanh²(z)

Démonstration (chain rule + quotient rule) :
    Posons u = e^z - e^-z et v = e^z + e^-z, donc tanh = u/v.
    u' = e^z + e^-z = v
    v' = e^z - e^-z = u
    tanh' = (u'v - uv') / v² = (v² - u²) / v² = 1 - (u/v)² = 1 - tanh²(z)

Avantage pratique : la dérivée se calcule à partir de tanh(z) lui-même,
pas besoin de recalculer les exponentielles. Si on stocke tanh(z) lors du
forward pass, le backward pass est gratuit.

**Pourquoi tanh plutôt que sigmoid ?**

Tanh est essentiellement une sigmoid "rescalée et recentrée" :
    tanh(z) = 2·sigmoid(2z) - 1

Différences pratiques :
1. Sortie centrée en 0 → meilleur conditionnement du gradient lors de la
   backprop. Avec sigmoid (sortie ∈ [0,1] donc toujours positive), les
   gradients qui arrivent sur les poids ont toujours le même signe, ce
   qui fait zigzaguer la descente. Tanh évite ce biais.
2. Dérivée max = 1 (en z=0) contre 0.25 pour sigmoid. Donc tanh atténue
   moins les gradients → vanishing gradient plus tardif.

**Limite : vanishing gradient quand même**

Pour |z| grand, tanh(z) sature à ±1, et tanh'(z) → 0. Le gradient devient
quasi-nul → les neurones saturés n'apprennent plus.
C'est pourquoi ReLU a largement remplacé tanh en deep learning moderne.
Tanh reste utile pour les RNN et certains cas spécifiques.

**Choix d'implémentation**

- On utilise `np.tanh(z)` plutôt que recoder la formule avec `np.exp` :
  l'implémentation numpy est numériquement stable (gère les grands |z|
  sans overflow) et écrite en C optimisé.
- Pour la dérivée, on recalcule `np.tanh(z)` au lieu de demander à l'appelant
  de fournir la valeur déjà calculée. C'est le contrat des tests et c'est
  cohérent avec relu_derivative qui prend aussi z en entrée.

### 1.3 Logistic (sigmoid)

**Définition mathématique**

σ(z) = 1 / (1 + e^-z)

Propriétés :
- Bornée : σ(z) ∈ ]0, 1[
- Croissante, dérivable partout
- Symétrie : σ(-z) = 1 - σ(z)
- σ(0) = 0.5

**Dérivée**

σ'(z) = σ(z) · (1 - σ(z))

Démonstration :
    σ(z) = (1 + e^-z)^-1
    σ'(z) = -1 · (1 + e^-z)^-2 · (-e^-z)
          = e^-z / (1 + e^-z)²
          = [1/(1+e^-z)] · [e^-z / (1+e^-z)]
          = σ(z) · [(1+e^-z - 1) / (1+e^-z)]
          = σ(z) · (1 - σ(z))

**Pourquoi sigmoid ?**

Usage principal : activation de la couche de SORTIE en classification binaire.
La sortie σ(z) ∈ ]0,1[ s'interprète comme P(y=1 | x). Le seuil 0.5
sépare les deux classes.

En activation cachée, elle a été abandonnée au profit de ReLU à cause de :
1. Vanishing gradient : σ'(z) max = 0.25 (atteint en z=0). Donc à chaque
   couche le gradient est divisé par au moins 4 → sur 10 couches,
   facteur 4^10 ≈ 10⁻⁶ : le gradient s'éteint.
2. Sortie non centrée (toujours > 0) → biais systématique sur les
   gradients de la couche suivante (tous les gradients ont le même signe
   → la descente zigzague).

**Piège numérique : overflow dans exp**

Pour z très négatif (ex. z = -1000), e^-z = e^1000 ≈ 10^434.
Hors de la plage des floats 64 bits (max ≈ 10^308) → overflow → inf → NaN.

Solution : clipper z à [-500, +500] avant de calculer exp.
À ±500, σ(z) est numériquement indistinguable de 0 ou 1 (différence < 10^-217),
donc le clip n'introduit aucune erreur perceptible.

**Lien avec tanh**

tanh(z) = 2·σ(2z) - 1

Donc tanh = sigmoid "recentrée et rescalée". Tanh corrige le défaut
de centrage de sigmoid, ce qui en fait un meilleur choix en cachée.
Mais en sortie binaire, sigmoid reste irremplaçable car la plage [0,1]
est naturellement interprétable comme une probabilité.

**Choix d'implémentation**

- `np.clip(z, -500, 500)` avant `np.exp(-z)` pour éviter l'overflow.
- La dérivée appelle `logistic(z)` pour réutiliser le clipping (DRY).
- On recalcule σ(z) dans la dérivée plutôt que de demander à l'appelant
  de fournir σ(z) déjà calculé — cohérent avec relu_derivative et tanh_derivative.

### 1.4 Softmax

**Définition mathématique**

Softmax prend un vecteur z = [z₁, z₂, ..., zₖ] (un score par classe)
et le transforme en distribution de probabilité :

    softmax(z)ᵢ = e^zᵢ / Σⱼ e^zⱼ

Propriétés :
- Chaque sortie ∈ ]0, 1[
- Σᵢ softmax(z)ᵢ = 1 par construction
- Strictement croissante en zᵢ (à autres composantes fixées)
- Plus zᵢ est grand par rapport aux autres, plus softmax(z)ᵢ → 1

**Usage**

Activation de la couche de SORTIE en classification multi-classe (K ≥ 3).
La sortie s'interprète comme P(y = classe k | x).
Avec K = 2, softmax est strictement équivalente à sigmoid.

**Invariance par translation (le point clé)**

Pour toute constante c :

    softmax(z - c) = softmax(z)

Démonstration :
    softmax(z - c)ᵢ = e^(zᵢ-c) / Σⱼ e^(zⱼ-c)
                    = (e^zᵢ · e^-c) / (e^-c · Σⱼ e^zⱼ)
                    = e^zᵢ / Σⱼ e^zⱼ
                    = softmax(z)ᵢ

Le facteur e^-c se simplifie au numérateur et au dénominateur.

**Piège numérique : overflow dans exp**

Pour zᵢ grand (ex. zᵢ = 1000), e^zᵢ ≈ 10^434, hors plage float64 (max ~10^308).
Overflow → inf → NaN dans la division.

Solution : exploiter l'invariance pour remplacer z par z - max(z).
Après soustraction, le max vaut 0, donc tous les e^... ≤ 1. Plus jamais
d'overflow. Le sous-flow (e^... très petit) ne cause pas de NaN, juste
des valeurs proches de 0, ce qui est numériquement acceptable.

**Pourquoi pas de softmax_derivative séparée ?**

En classification multi-classe, softmax est TOUJOURS couplée à
cross-entropy comme loss. La dérivée combinée de
L = cross_entropy(softmax(z), y_true) par rapport à z se simplifie
spectaculairement :

    ∂L/∂z = ŷ - y_true

où ŷ = softmax(z) et y_true est l'encodage one-hot.

C'est la même forme que la dérivée de (sigmoid + binary cross-entropy)
en binaire, et que la dérivée de (linear + MSE) en régression. Cette
simplification universelle n'est PAS un hasard : c'est une propriété des
"matched pairs" loss/activation issues du cadre des modèles linéaires
généralisés (GLM).

Conséquence pratique : pas besoin de coder softmax_derivative. On
implémentera directement (ŷ - y) dans le backward du classifier (Phase 4).

**Détails d'implémentation pour le batch**

Entrée z de shape (n_samples, n_classes) → un sample par ligne, traité
indépendamment.

- `np.max(z, axis=1, keepdims=True)` : max par ligne, shape (n_samples, 1)
  pour préserver le broadcasting.
- Le broadcasting numpy étire automatiquement (n_samples, 1) vers
  (n_samples, n_classes) lors de la soustraction → chaque ligne est
  shifted par son propre max.
- Idem pour `np.sum(axis=1, keepdims=True)` au dénominateur.

Sans `keepdims=True`, la shape serait (n_samples,) au lieu de (n_samples, 1),
le broadcasting échouerait et numpy lèverait une erreur.

**Lien avec sigmoid**

En binaire (K=2), avec z = [z₁, z₂] :
    softmax(z)₁ = e^z₁ / (e^z₁ + e^z₂) = 1 / (1 + e^(z₂-z₁)) = σ(z₁ - z₂)

Donc softmax binaire = sigmoid d'une différence. C'est pour ça qu'on dit
que softmax "généralise" sigmoid au cas multi-classe.

## Phase 2 — Initialisation des poids

### Les 4 paramètres à initialiser

Pour un réseau à une couche cachée :
- W₁ : shape (n_features, hidden_layer_size) — poids input → hidden
- b₁ : shape (hidden_layer_size,) — biais de la hidden layer
- W₂ : shape (hidden_layer_size, n_outputs) — poids hidden → output
- b₂ : shape (n_outputs,) — biais de la output layer

Total à apprendre : (n_features × H) + H + (H × n_outputs) + n_outputs paramètres.
Pour breast cancer (30 features, H=30, binaire) : 30×30 + 30 + 30×1 + 1 = 961 paramètres.

### Pourquoi pas zéro ? Le problème de symétrie

Si W₁ = 0 partout, alors :
    z₁ = X @ 0 + 0 = 0 pour tous les neurones cachés
    a₁ = activation(0) = constante identique pour tous les neurones

Tous les neurones produisent la même valeur. Pendant la backprop, ils
reçoivent tous le même gradient (par symétrie), donc sont mis à jour de
manière identique. Itération après itération, ils restent identiques.

Conséquence : un réseau à 100 neurones cachés initialisés à 0 se comporte
comme un réseau à 1 seul neurone caché. On perd toute la capacité du modèle.

### Pourquoi des valeurs ALÉATOIRES casse la symétrie

Chaque neurone reçoit une initialisation différente → produit une valeur
différente → reçoit un gradient différent → évolue différemment.
Les neurones se spécialisent progressivement sur des features différentes
des données.

### Pourquoi un PETIT écart-type (0.01) ?

Compromis entre deux extrêmes :
- σ trop grand → z = X·W est grand en valeur absolue → activations saturent
  (sigmoid et tanh donnent ≈ 0 ou ≈ 1 → dérivée ≈ 0 → vanishing gradient
  dès l'itération 1, le réseau n'apprend pas).
- σ trop petit → z trop proche de 0 → activations toutes très proches → 
  signaux trop similaires entre neurones → apprentissage lent.

σ = 0.01 est une valeur "raisonnable par défaut" recommandée par le brief.
Il existe des schémas d'init plus sophistiqués (Xavier/Glorot, He) qui
adaptent σ à la taille des couches pour un meilleur conditionnement,
mais ils ne sont pas demandés ici.

### Pourquoi les biais peuvent rester à zéro

Le problème de symétrie ne concerne QUE les poids W. Les biais sont juste
des décalages additifs : tant que les W sont différents, les neurones se
différencient même si tous les b commencent à 0. Convention standard.

### Reproductibilité (random_state)

np.random.default_rng(seed) crée un générateur déterministe : avec le
même seed, on obtient toujours la même séquence de nombres aléatoires.
Indispensable pour :
- déboguer (résultats reproductibles)
- comparer deux modèles dans des conditions équivalentes
- faire passer le test test_reproducibility

Si seed=None, le générateur est initialisé avec une source d'entropie système
(horloge, etc.) → résultats différents à chaque exécution.

## Phase 3 — Implémentation du Régresseur

### 3.1 Forward propagation

**Équations du réseau (one hidden layer, regression)**

    z₁ = X @ W₁ + b₁              (combinaison linéaire couche cachée)
    a₁ = σ(z₁)                    (activation non-linéaire)
    z₂ = a₁ @ W₂ + b₂             (combinaison linéaire sortie)
    ŷ  = z₂                       (sortie = identité, pour régression)

**Shapes**

    X     : (n_samples, n_features)
    W₁    : (n_features, H)        H = hidden_layer_size
    b₁    : (H,)
    z₁,a₁ : (n_samples, H)
    W₂    : (H, n_outputs)
    b₂    : (n_outputs,)
    z₂,ŷ  : (n_samples, n_outputs)

**Broadcasting du biais**

X @ W₁ a shape (n_samples, H). b₁ a shape (H,). Numpy ajoute b₁ à
chaque ligne automatiquement (broadcasting). Équivalent à dupliquer
b₁ verticalement n_samples fois, mais sans coût mémoire.

**Pourquoi retourner z₁, a₁, z₂, ŷ (et pas seulement ŷ) ?**

La backpropagation a besoin de tous les états intermédiaires pour
calculer les gradients (chain rule). On les mémorise au forward pass
plutôt que de les recalculer. Trade-off mémoire/temps : standard en DL.

**Pourquoi pas d'activation en sortie pour la régression ?**

On veut prédire un nombre réel ŷ ∈ ℝ. Toute activation bornée (sigmoid,
tanh) limiterait la plage de sortie. L'identité (= pas d'activation)
laisse z₂ tel quel → ŷ peut prendre n'importe quelle valeur réelle,
adapté à la plupart des cibles continues.

Note : si la cible est positive (ex. prix), on pourrait utiliser ReLU
ou softplus en sortie. Mais identity reste le défaut sklearn-compatible.


### 3.2 Loss : Mean Squared Error

**Définition**

    MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

Propriétés :
- ≥ 0 toujours, = 0 ssi prédiction parfaite
- Pénalise quadratiquement → outliers ont fort impact
- Différentiable partout (clé pour gradient descent)

**Pourquoi quadratique et pas |.|**

1. Différentiabilité : |x| a un point anguleux en 0, x² est lisse partout.
2. Outliers : pénalisation forte des grosses erreurs → corrige en priorité.
3. Solution analytique connue : MSE est la log-vraisemblance d'un modèle
   gaussien (bruit normal), liaison avec la statistique classique.

Alternative : MAE = (1/n)Σ|yᵢ - ŷᵢ|, plus robuste aux outliers mais
non différentiable en 0. Non demandé ici.

**Pourquoi cette loss correspond à la sortie identité**

Comme softmax+cross_entropy ou sigmoid+binary_cross_entropy, le couple
identity+MSE est un "matched pair" issu des GLM (modèles linéaires
généralisés). Sa dérivée combinée se simplifie en :

    ∂L/∂z₂ = ŷ - y_true

C'est exactement la même forme que pour les autres matched pairs.
Conséquence : la backprop sera élégante (Phase 3.3).

### 3.3 Backpropagation

**Objectif**

Calculer ∂L/∂W₁, ∂L/∂b₁, ∂L/∂W₂, ∂L/∂b₂ — les 4 gradients qui
indiquent comment modifier chaque paramètre pour faire baisser la loss.

**Outil mathématique : chain rule**

Si y = f(g(x)), alors dy/dx = f'(g(x)) · g'(x).
Dans le réseau, la loss dépend de z₂ qui dépend de W₂ et a₁,
qui dépend de z₁ qui dépend de W₁. On applique la chain rule
en remontant cette chaîne depuis la sortie.

**Démonstration étape par étape (régression, identity + MSE)**

Notations : n = nb_samples, L = (1/n) Σ (yᵢ - ŷᵢ)², ŷ = z₂.

(1) Gradient à la couche de sortie

    ∂L/∂z₂ = ∂/∂z₂ [(1/n) Σ (y - z₂)²]
           = (1/n) · 2 · (z₂ - y) · (-(-1))
           = (2/n) · (ŷ - y)

    Le facteur 2 est constant : on l'absorbe dans le learning rate.
    En pratique on utilise :
        dz₂ = (1/n) · (ŷ - y)

    Note : c'est la même forme que pour (sigmoid + binary_CE) en classif
    binaire et (softmax + CE) en multi-classe. Coïncidence apparente,
    propriété profonde des "matched pairs" loss/activation (cadre GLM).

(2) Gradients de la couche de sortie

    z₂ = a₁ W₂ + b₂   →  linéaire en W₂ et b₂

    dW₂ = a₁ᵀ · dz₂        (shape : (H, n_outputs))
    db₂ = Σ_samples dz₂    (somme sur axis=0)

    Pourquoi la transposée ? La règle matricielle dit que si y = A·x,
    alors ∂y/∂A = x · (extérieurement). En batchant sur n samples,
    on accumule via aᵀ·dz.

    Pourquoi sommer pour db₂ ? b₂ est PARTAGÉ entre les n samples
    (même biais pour tout le batch). Donc son gradient = somme des
    contributions de chaque sample.

(3) Gradient propagé à la couche cachée

    a₁ = σ(z₁), z₂ = a₁ W₂ + b₂

    da₁ = dz₂ · W₂ᵀ                 (chain rule, shape (n, H))
    dz₁ = da₁ ⊙ σ'(z₁)              (⊙ = multiplication élément par élément)

    Pourquoi élément par élément ? σ agit composante par composante
    sur z₁, donc sa dérivée aussi : (σ(z))ᵢ ne dépend que de zᵢ.
    Matriciellement, dσ/dz est une matrice diagonale, dont le produit
    avec un vecteur revient à une multiplication composante par composante.

(4) Gradients de la couche cachée

    z₁ = X W₁ + b₁   →  symétrique à l'étape (2)

    dW₁ = Xᵀ · dz₁    (shape : (n_features, H))
    db₁ = Σ_samples dz₁

**Pattern à retenir**

Pour chaque couche, on calcule dz (la "responsabilité" de cette couche
dans l'erreur), puis localement dW et db, puis on propage à la couche
précédente via dz_prev = dz · Wᵀ ⊙ σ'(z_prev).

C'est ce pattern qu'on appellera "backward propagation" — le gradient
"se propage" de la sortie vers l'entrée, en sens inverse du forward pass.

**Vérification des shapes (essentiel pour débugger)**

    X      : (n, n_features)
    W₁     : (n_features, H)        Xᵀ : (n_features, n)
    z₁,a₁  : (n, H)
    W₂     : (H, n_outputs)         W₂ᵀ : (n_outputs, H)
    z₂,ŷ,y : (n, n_outputs)
    
    dz₂    : (n, n_outputs)         comme z₂ ✓
    dW₂    : (H, n_outputs)         a₁ᵀ @ dz₂ = (H,n)@(n,n_out) ✓
    db₂    : (n_outputs,)           sum(dz₂, axis=0) ✓
    dz₁    : (n, H)                 (n,n_out)@(n_out,H) = (n,H) ✓
    dW₁    : (n_features, H)        Xᵀ @ dz₁ = (n_feat,n)@(n,H) ✓
    db₁    : (H,)                   sum(dz₁, axis=0) ✓

Astuce de soutenance : si le prof demande "comment t'assures-tu que
les gradients ont la bonne shape ?", tu réponds que les gradients ont
TOUJOURS la même shape que le paramètre qu'ils dérivent (dW₁ a shape
de W₁, etc.) — et tu pointes le tableau ci-dessus.

**Choix d'implémentation**

- On divise dz₂ par n_samples ICI (et pas dans la loss) pour avoir
  des gradients normalisés. Conséquence : on n'a pas besoin de moyenner
  ailleurs.
- On utilise @ pour les produits matriciels (lisible) et * pour
  l'élément par élément. Erreur courante : confondre les deux.
- np.sum(..., axis=0) : axis=0 = sommer le long des lignes, donc on
  obtient un vecteur de longueur égale au nombre de colonnes — exactement
  ce qu'il faut pour le biais.

### 3.4 La méthode fit() — learning loop

**Pseudocode**

    fit(X, y):
        normaliser y en shape (n, n_outputs)
        initialiser W₁, b₁, W₂, b₂
        loss_curve = []
        répéter max_iter fois :
            forward → z₁, a₁, z₂, ŷ
            loss = MSE(y, ŷ)
            loss_curve.append(loss)
            backward → dW₁, db₁, dW₂, db₂
            W₁ ← W₁ - η · dW₁
            b₁ ← b₁ - η · db₁
            W₂ ← W₂ - η · dW₂
            b₂ ← b₂ - η · db₂
        return self

**Gradient descent : la règle de mise à jour**

    W ← W - η · ∂L/∂W

Intuition géométrique : ∂L/∂W pointe dans la direction de plus forte
AUGMENTATION de la loss. Pour la DIMINUER, on va dans la direction
opposée → signe moins. η contrôle la longueur du pas.

Choix de η :
- Trop grand → on saute par-dessus le minimum, la loss oscille ou diverge
- Trop petit → on converge correctement mais très lentement
- En pratique : 1e-3 à 1e-2 marche bien pour ce projet (cf. tests)

**Pourquoi reshape y en 2D ?**

La backprop calcule dz₂ = (ŷ - y) / n. Si ŷ a shape (n, 1) et y a
shape (n,), numpy fait du broadcasting qui crée une shape (n, n) —
résultat catastrophique. Forcer y à (n, 1) garantit que la soustraction
est élément par élément, ligne par ligne.

**Pourquoi convertir en float ?**

Si y arrive en int (ex: targets entiers), numpy fait des opérations
entières et arrondit les gradients à 0 → le réseau n'apprend rien.
np.asarray(y, dtype=float) force la conversion.

**Pourquoi réinitialiser loss_curve_ = [] à chaque fit() ?**

Sinon, si l'utilisateur fait model.fit(X1, y1) puis model.fit(X2, y2),
les courbes des deux trainings seraient concaténées. On veut une
courbe propre par appel à fit().

**Mode de descente : full-batch (vs mini-batch / SGD)**

Notre implémentation utilise X **entier** à chaque itération
(full-batch gradient descent). Avantages :
- Gradient exact (pas de bruit stochastique)
- Implémentation simple, code court
- Bon comportement sur petits datasets (le projet)

Inconvénients (pour info, pas implémenté ici) :
- Coûteux en mémoire sur très gros datasets
- Plus susceptible de tomber dans un minimum local (le bruit du SGD
  peut au contraire aider à en sortir)

Le brief mentionne mini-batch et SGD dans les "Advanced requirements"
mais ne l'impose pas. On reste full-batch pour la simplicité.

**Pourquoi return self ?**

Convention scikit-learn : permet le chaînage
    model = SimpleSLPRegressor().fit(X, y)
au lieu de
    model = SimpleSLPRegressor()
    model.fit(X, y)
Test test_reproducibility exploite cette convention.

### 3.5 predict() et score()

**predict() : forward pass sans apprentissage**

Une fois fit() terminé, les poids W₁, b₁, W₂, b₂ sont fixés. predict()
réutilise _forward_propagation(X) avec ces poids gelés. On ignore les
intermédiaires (z₁, a₁, z₂) puisqu'on n'a pas besoin de la backprop.

Reshape de sortie : si n_outputs = 1, on aplatit en 1D avec .ravel()
pour suivre la convention sklearn (y_pred 1D pour single-target).
Ce détail fait passer le test test_predict_shape_and_finite.

**score() : le R² (coefficient de détermination)**

Définition :

    R² = 1 - SS_res / SS_tot
    
    SS_res = Σᵢ (yᵢ - ŷᵢ)²       (résidus du modèle)
    SS_tot = Σᵢ (yᵢ - ȳ)²        (variance totale des y)

Interprétation : proportion de la variance des y expliquée par le modèle.
- R² = 1 → prédiction parfaite (SS_res = 0)
- R² = 0 → le modèle équivaut à prédire la moyenne ȳ partout
- R² < 0 → le modèle est PIRE qu'une prédiction constante = ȳ
           (arrive sur test set difficile, modèle sous-entraîné, ou
            quand y_train et y_test ont des distributions différentes)

**Pourquoi R² plutôt que MSE comme métrique ?**

MSE dépend de l'échelle des y : MSE = 100 n'a aucun sens absolu.
- Sur prix d'avocats (1€-3€) : MSE=100 = catastrophe
- Sur prix de maisons (100k€-1M€) : MSE=100 = excellent

R² est ADIMENSIONNEL → comparable entre datasets. Convention sklearn :
.score() retourne R² pour les régresseurs, accuracy pour les classifieurs.

**Démonstration du cas R² = 0 (intéressant pour la soutenance)**

Si on prédit ŷᵢ = ȳ pour tout i (modèle constant qui ignore X), alors :
    SS_res = Σ (yᵢ - ȳ)² = SS_tot
    R² = 1 - SS_tot/SS_tot = 0

Donc R² = 0 c'est le baseline naïf. Un modèle utile DOIT faire mieux.

**Pourquoi reproductibilité dans predict() ?**

predict() est déterministe : pas de tirage aléatoire, pas d'init.
Pour deux modèles entraînés avec le même random_state sur les mêmes
données, predict(X_test) renvoie EXACTEMENT le même résultat — ce que
vérifie test_reproducibility.

**Choix d'implémentation**

- predict() délègue à _forward_propagation() pour ne pas dupliquer
  la math du forward.
- score() délègue à predict() pour la même raison (DRY).
- Pas de gestion d'erreur "modèle non entraîné" — on suppose que
  l'utilisateur appelle fit() avant predict(). Sinon W1_ vaut None
  et numpy lèvera une AttributeError explicite (None has no attribute T).

### Phase 3 failed modifications 

### Choix du défaut d'activation : relu plutôt que logistic

Le scaffold du prof propose `activation: str = "logistic"` par défaut.
On a remplacé par `"relu"` pour le régresseur. Justification :

- Logistic en cachée borne a₁ ∈ [0,1] → z₂ = a₁W₂ + b₂ borné par
  H·max(W₂) + b₂. Avec init σ=0.01 et H=20, sortie initiale ≈ 0.
- Cibles régression peuvent être grandes (|y| ~ 10 dans test_linear_function).
  Pour atteindre |ŷ| ~ 10 avec a₁ ≤ 1, il faut W₂ ~ 0.5 — partant de 0.01,
  ça prend de très nombreuses itérations.
- ReLU n'est pas bornée, dérivée = 1 pour z > 0 → convergence directe.
- Alignement avec sklearn.MLPRegressor (default relu également).

Le test test_linear_function (R² > 0.8) échouait avec logistic mais passe
avec relu — illustration concrète de l'importance du choix d'activation
pour la régression sur cibles à grande amplitude.

### Choix du défaut de learning_rate : 0.01 plutôt que 0.001

Même logique que pour l'activation : le scaffold reprenait les défauts
sklearn (lr=0.001) qui sont calibrés pour le solveur adam (adaptatif).
Notre implémentation utilisant un gradient descent vanille (full-batch SGD),
0.001 est trop petit pour converger en max_iter=200 itérations dans le
cas où la cible a une magnitude importante OU une moyenne non nulle.

Exemple : test_constant_output (y=5 partout, max_iter=100).
Avec lr=0.001, le réseau converge vers ŷ≈3.2 — bien mais insuffisant.
Avec lr=0.01, ŷ atteint 5 dans la fenêtre de temps imposée.

Trade-off : lr trop grand → divergence ou oscillation. lr=0.01 est un
sweet spot empirique vérifié par les tests fournis.

## Choix de l'écart-type d'initialisation : σ = 0.15

Le brief suggère 0.01 en exemple, mais c'est trop petit pour ReLU :
trop de neurones démarrent inactifs (z ≤ 0) → "dying ReLU" →
convergence stagnante.

Référence théorique : He initialization recommande σ_He = sqrt(2/n_features)
pour ReLU. Donne :
- n_features = 3  → σ_He ≈ 0.82
- n_features = 10 → σ_He ≈ 0.45
- n_features = 30 → σ_He ≈ 0.26

Notre choix σ = 0.15 est un compromis conservateur : assez grand pour
éviter le dying ReLU sur les petits réseaux des tests (n_features = 3),
assez petit pour ne pas saturer tanh/sigmoid sur les grands réseaux.

Alternative plus rigoureuse (non implémentée pour rester simple) :
adapter σ à n_features selon He pour ReLU, Xavier pour tanh/sigmoid.
Notre σ unique passe tous les tests sans complexifier le code.