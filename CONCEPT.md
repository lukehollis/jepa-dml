CONFIDENTIAL // DO NOT DISTRIBUTE WITHOUT PRIOR CONSENT
The Everything Sim
Global Causal Engine for Mission-Critical Intelligence Autonomy
Luke Hollis
Research Scientist, MIT, Harvard
lhollis@g.harvard.edu
Abstract
Current intelligence analysis relies on slow, correlation-based methods that ignore
causality amid complex confounders. We propose a global causal inference engine
that uses a self-supervised predictive architecture (JEPA) world model to learn
the underlying causes of events from unstructured, high-dimensional data. This
enables human decision-makers as well as autonomous systems to conduct robust
analysis with counterfactuals and parameterized models.
Existing Limitations in Intelligence Analysis
Human intelligence analysts across defense and industry largely do forecasting via correlational
methods and are constrained due to the limitations of experiment design, inability to track variables
across diverse, multimodal datasets, and a general lack of knowledge of causality throughout their
organizations.
We developed a world model to act as a "deconfounder" to conduct causal analysis in high dimensional
space after encoding diverse data inputs in a lower dimensional field. We take advantage of existing
language models for assisting non-technical users with experiment design, onboarding diverse
intelligence datasets and selecting from our own signals, geospatial, and open source intelligence
datasets. We then create robust, custom parameterized models that analysts can use to provide rapid,
actionable intelligence products.
World Model as Deconfounder
To accomplish this, Luke Hollis (Harvard/MIT) presents a framework for causal inference from
high-dimensional data that addresses latent confounders in dynamic environments. In observational
intelligence streams, an unobserved variable U often resides in the temporal history, influencing
both treatment T and outcome Y . This creates a confounding pathway (T ←U →Y ) that renders
standard causal identification impossible on non-IID data.
We employ a spatiotemporal Joint Embedding Predictive Architecture (JEPA) to learn a low-
dimensional representation R. By encoding state trajectories rather than static snapshots, we adapt
state estimation techniques from autonomous robotics to restore the Markov property and satisfy
the sufficiency assumption in non-IID streams. This ensures R captures historical dependencies,
blocking the confounding path and enabling identification of P (Y |do(T )). This representation is
integrated into a Double Machine Learning (DML) framework to power reliable decision-making
under uncertainty. Benchmarks on semi-synthetic data show our JEPA-DML reduces bias (e.g., by
up to 86%) over baselines like DragonNet.
Preprint.
CONFIDENTIAL // DO NOT DISTRIBUTE WITHOUT PRIOR CONSENT
CONFIDENTIAL // DO NOT DISTRIBUTE WITHOUT PRIOR CONSENT
Traction and Dual-use GTM
Right now we have been entirely focused on the Department of War, where we’ve seen firsthand the
need for rapid response planning in contested environments. Our Causal Engine enables operators
to ask "what-if" questions and simulate thousands of potential actions, accelerating their planning
cycles. We’ve gained traction with deployments in defense units and are applying to SBIR and other
funding sources.
This success has also attracted investors, who have onboarded custom datasets to query their data,
revealing similar opportunities for serving McKinsey, Bain, or buyers of their intelligence products.
The exploratory founding team brings expertise in defense and intelligence from McKinsey, Bain,
and related companies with knowledge of existing buyers and deliverables.
Leveraging this defense validation, we will expand the same core technology to commercial verti-
cals like finance, insurance, and supply chain management for modeling disruptions among other
scenarios.
Causality, Inc
We believe the future of intelligence work is a global causal engine that can rapidly adapt to
multimodal data and describe causal relationships. We are seeking partners in raising a Seed round
to scale our execution based on our belief in this future. The funds will be used for building our
engineering teams, GTM leadership, and training our foundational causal models.
We will continually be dogfooding our startup’s direction by using our own tool for running causal
analysis of our product roadmap and opportunities for building the future of intelligence work and
autonomous systems.
LUKE HOLLIS | lhollis@g.harvard.edu | +1 617.372.4811
CONFIDENTIAL // DO NOT DISTRIBUTE WITHOUT PRIOR CONSENT

UPDATE 

- frontend with LLM guided experiment design is /Users/lrh/Projects/causality/causality
- world model jepa-dml is /Users/lrh/Projects/causality/jepa_dml