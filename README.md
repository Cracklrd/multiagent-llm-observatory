# Observatoire interactif d'un systeme multiagents IA

Projet de classe visant a observer les couts, delais, jetons, erreurs et resultats d'un systeme multiagents pour optimiser des applications LLM.

## Phase actuelle

Les phases 1 a 3 mettent en place la fondation, les outils et le systeme multiagents :

- structure des dossiers ;
- configuration locale ;
- schema SQLite ;
- modeles de donnees Python ;
- script d'initialisation de la base ;
- recherche et comparaison d'un catalogue de modeles simules ;
- calcul reproductible du cout par million de jetons ;
- evaluation explicable de la qualite par couverture de mots-cles ;
- cinq agents CrewAI relies dans un processus sequentiel ;
- execution OpenAI en mode live et repli local en mode demo ;
- tests automatises.

Le tracing complet, les scenarios et le dashboard d'observabilite seront ajoutes dans les phases suivantes.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env
```

`DEMO_MODE=true` permet de garder le projet executable avec des donnees locales sans cle API obligatoire.

## Initialisation

```bash
python scripts/init_db.py
```

## Tests

```bash
pytest
```

## Application locale

```bash
streamlit run app.py
```

Ouvrir ensuite `http://localhost:8501` dans le navigateur.

## Exemple des outils

```python
from tools import calculate_cost, evaluate_quality, recommend_model

model = recommend_model("balanced")
cost = calculate_cost(model.model_name, input_tokens=2_000, output_tokens=500)
quality = evaluate_quality(["cout", "latence"], "Le cout et la latence sont mesures.")
```

Les prix de `data/models.csv` sont simules et exprimes par million de jetons. Les memes donnees locales sont donc utilisables sans cle API pendant une demonstration.

## Execution des agents

Le mode local ne consomme aucun jeton :

```bash
python scripts/run_crew.py
```

Pour utiliser OpenAI, definir `OPENAI_API_KEY` dans `.env`, mettre `DEMO_MODE=false`, puis lancer :

```bash
python scripts/run_crew.py --live
```

Le modele OpenAI par defaut est `gpt-4.1-mini` et peut etre configure avec `OBSERVATORY_OPENAI_MODEL`. Les modeles `model_fast`, `model_balanced` et autres restent des profils simules que les agents doivent analyser ; ils ne sont pas les modeles qui generent les reponses.
