# Research Plan: World Models and World Action Models

## Root Directory Tree
```
deep-dive-world-model-world-action-model/
├── README.md
├── 01-overview.md
├── 02-history.md
├── 03-world-models-principles.md
├── 04-world-action-models-principles.md
├── 05-engineering-implementation.md
├── 06-applications-use-cases.md
├── 07-summary-recommendations.md
└── PROGRESS.md
```

## Document Details

### README.md
- **Title**: Reading Guide: World Models and World Action Models
- **Core Coverage**: 
  - Explain the relationships between documents
  - Suggested learning path (start with overview, then history, then principles, etc.)
  - Define key terminology: world model, world action model, VLA, etc.
  - List prerequisites (basic RL, probability, neural networks)

### 01-overview.md
- **Title**: System Architecture and High-Level Decomposition
- **Core Coverage**:
  - High-level decomposition of world models and world action models
  - Module relationship diagrams (perception, world model, action model, planning, control)
  - Responsibilities and interactions of each part
  - Comparison with traditional model-based RL and end-to-end learning

### 02-history.md
- **Title**: History and Context
- **Core Coverage**:
  - Trace the development history of world models (from Ha & Schmidhuber 2018 World Models, to DreamerV2/V3, etc.)
  - Evolution of action-conditioned world models and world action models
  - Important milestones: introduction of VLA models, integration with robotics
  - Relationship and differences with existing algorithms (MBRL, imitation learning, etc.)

### 03-world-models-principles.md
- **Title**: Principles and Design of World Models
- **Core Coverage**:
  - Mathematical foundations: state space models, latent dynamics, observation models
  - Types of world models: latent (Dreamer), generative (video prediction), probabilistic (Semantic Bayesian WM)
  - Detailed mathematical derivations with variable mappings
  - Engineering considerations: training objectives, stability, scalability
  - Pseudocode and key code snippets

### 04-world-action-models-principles.md
- **Title**: Principles and Design of World Action Models
- **Core Coverage**:
  - Definition: models that predict both world states and actions, or integrate action prediction
  - Architectures: direct-action VLA policies, World-Action Models (WAM), inverse-dynamics WAM
  - Mathematical formulations: joint prediction of states and actions, action-conditioned dynamics
  - Training paradigms: supervised learning from demonstrations, RL fine-tuning
  - Variable mapping tables linking math to code

### 05-engineering-implementation.md
- **Title**: Engineering Implementation and Runnable Examples
- **Core Coverage**:
  - Minimal reproducible examples (MRE) for world models (e.g., simple latent dynamics in PyTorch)
  - Code snippets for world action models (e.g., predicting actions from latent states)
  - Engineering deployment points: real-time inference, hardware considerations
  - Framework version requirements (PyTorch, TensorFlow, etc.)
  - Common configurations and hyperparameters

### 06-applications-use-cases.md
- **Title**: Examples and Use Cases
- **Core Coverage**:
  - Robotics: manipulation, locomotion, loco-manipulation (as seen in GIFT, FWBC-VLA)
  - Autonomous driving: simulation-based planning
  - Video generation: controllable video generation with action conditioning
  - Decision-making: safety-critical systems (Risk-Informed WM)
  - Common configurations and best practices

### 07-summary-recommendations.md
- **Title**: Summary and Recommendations
- **Core Coverage**:
  - Brief summary of applicable scenarios for each approach
  - Performance bottlenecks and computational trade-offs
  - Future directions: unified architectures, sim-to-real transfer, safety guarantees
  - Engineering practice recommendations: when to use world models vs. world action models

### PROGRESS.md
- Checklist tracking completion of each document (⬜ Pending → 🔄 In Progress → ✅ Done)

## Key Questions List
1. What are the core mathematical differences between latent world models and generative world models?
2. How do world action models integrate action prediction with world state prediction, and what are the benefits over separate models?
3. What are the key architectural innovations in recent VLA and WAM models (e.g., GIFT, WorldReward)?
4. How are world models trained and evaluated in practice, especially for robotic applications?
5. What are the main challenges in deploying world models and world action models in real-world systems (sim-to-real gap, computational latency, safety)?
6. How do probabilistic world models (like Semantic Bayesian WM) handle uncertainty and decision-making?
7. What role do world models play in enabling long-horizon planning and reasoning in embodied AI?
8. What are the current benchmarks and evaluation protocols for world models and world action models?
9. How can world models be made more sample-efficient and computationally efficient?
10. What are the emerging trends towards unified perception-action-reasoning systems in robotics?

## Notes
- All documents will be written in Chinese, following the user's language preference.
- Technical terms and paper titles will remain in English, with optional Chinese translations on first occurrence.
- Each major document will include Lab Exercises for hands-on verification.
- Citations will be provided for key claims, with links to arXiv papers, project pages, etc.