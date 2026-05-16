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