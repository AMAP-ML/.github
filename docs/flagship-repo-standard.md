# AMAP-ML Flagship Repository Standard

This checklist is for repositories that represent AMAP-ML publicly. A flagship repository should be understandable in 30 seconds, runnable in 30 minutes, and credible enough for researchers to cite, reproduce, and extend.

AMAP-ML's public narrative should present the team as a product-facing AI team that connects research, engineering, and real-world deployment. The homepage and pinned repositories should make two product anchors visible:

- Spatial intelligence: agents, spatial reasoning, route planning, world models, and product-scale mobility systems.
- Generative intelligence: image, video, 3D, text rendering, editing, visual effects, quality evaluation, virtual try-on, and scalable AIGC systems for AMAP map-native content assets, media, and user experiences.

The homepage should also preserve high-density public signals from the upstream profile:

- Quantitative highlights such as publication counts, open-source project count, and venue breadth.
- Recent launches from `origin/main`, especially `CoEvolve`, `DreamX-World`, `MACE-Dance`, `FASA`, `Eevee`, and `Taming-Hallucinations`.
- A folded update archive so the first screen remains focused while still retaining public activity history.
- Contribution-first descriptions. Venue labels should support the claim, not lead it.

## Recommended Pinned Repositories

Pin 4 to 6 repositories that best express the AMAP-ML portfolio. The pinned set should include spatial intelligence, generative intelligence, and core AI systems, while also respecting public adoption signals such as GitHub stars.

Default pinned set, balancing narrative coverage and public traction:

| Priority | Repository | Narrative role |
|:--|:--|:--|
| 1 | `SkillClaw` | Highest-traction core AI system: agentic skill evolution from real interaction traces. |
| 2 | `FluxText` | High-traction generative intelligence: scene-text editing and visual asset generation. |
| 3 | `Code2World` | High-traction core AI system: GUI world model and executable simulation. |
| 4 | `Tree-GRPO` | High-traction core RL method for agent reinforcement learning. |
| 5 | `GPG` | Core RL baseline for model reasoning with broader framework adoption. |
| 6 | `MobilityBench` | Spatial intelligence anchor: real-world route-planning agent evaluation. |

## Star Snapshot for Flagship Selection

This snapshot is used to balance public traction with narrative coverage. Recheck before changing pinned repositories because stars move quickly.

| Repository | Approx. stars | Primary role |
|:--|:--|:--|
| `SkillClaw` | 1.3k+ | Core AI system / agent infrastructure |
| `FluxText` | 448 | Generative intelligence |
| `Tree-GRPO` | 350 | Core AI system / agent RL |
| `Code2World` | 320 | Core AI system / world model |
| `FE2E` | 237 | Generative vision |
| `AutoDrive-R2` | 202 | Spatial intelligence / autonomous driving |
| `RL3DEdit` | 199 | Generative intelligence / 3D editing |
| `GPG` | 182 | Core AI system / reasoning RL |
| `Omni-Effects` | 175 | Generative intelligence / visual effects |
| `MobilityBench` | 136 | Spatial intelligence anchor |
| `Omni-WorldBench` | 107 | World-model evaluation |

Recent upstream projects that should be tracked in the next star refresh:

| Repository | Current role to monitor |
|:--|:--|
| `CoEvolve` | Agent-data mutual evolution; likely core agent-RL candidate if public traction grows. |
| `DreamX-World` | Interactive world model; already important narratively because it strengthens the world-model axis. |
| `MACE-Dance` | High-venue generative video project; useful for showing generative intelligence beyond image editing. |
| `FASA-ICLR2026` | Efficient attention / inference method; useful as a core systems supplement, but not a default homepage flagship yet. |
| `Eevee` | Video virtual try-on dataset and benchmark; useful when emphasizing product-facing generative media. |
| `Taming-Hallucinations` | Multimodal-video reliability; useful when emphasizing evaluation and model robustness. |

Optional rotation slots:

| Repository | Use when the homepage needs to emphasize |
|:--|:--|
| `Thinking-with-Map` | Map-augmented reasoning agent; promote when the homepage should emphasize the AMAP-specific spatial angle. |
| `FE2E` | High-traction generative vision project; rotate in when emphasizing vision/generation adoption. |
| `SpatialGenEval` | Spatial intelligence in generative models. |
| `DreamX-World` | Interactive world model for controllable world simulation. |
| `Omni-WorldBench` | Interactive 4D world-model evaluation. |
| `AutoDrive-R2` | Mobility-aligned VLA reasoning and self-reflection for autonomous driving. |
| `CoEvolve` | Agent-data mutual evolution; rotate in after stronger star traction or when agent training is the homepage emphasis. |
| `MACE-Dance` | Video generation and SIGGRAPH-level generative media signal. |
| `Eevee` | Product-facing virtual try-on and high-resolution generative video benchmark. |
| `DSFNet` | Industrial route ranking, public route dataset, and AMap deployment. |
| `IntTravel` | Industrial-scale travel recommendation data and deployed generative recommendation. |
| `EMF` | Text-conditioned one-step image generation. |
| `DCW` | Diffusion-model quality improvement for generative AI systems. |
| `RL3DEdit` | 3D generation and RL-based editing. |
| `Omni-Effects` | Controllable visual effects generation; rotate in when AIGC product relevance matters more than star ranking. |
| `Taming-Hallucinations` | Multimodal video reliability and counterfactual data generation. |

## Homepage Information Density

Keep the organization homepage dense, but not archival:

- First screen: team identity, product-facing foundation AI narrative, follow/join badges, and broad venue/open-source signal.
- Second section: `Highlights`, with compact quantitative proof points from the latest public profile.
- Third section: `What We Build`, organized by product anchors rather than paper areas.
- Fourth section: `Flagship Releases`, with 4 to 6 high-signal repositories chosen by narrative fit plus star traction.
- Later sections: `Recent Updates` and `Project Map`. Use `<details>` for older news so the homepage does not become a chronological paper feed.

Avoid these homepage patterns:

- Listing every paper before explaining what the team builds.
- Using the old DreamX-branded wording as the team identity.
- Mentioning unrelated AMAP teams or collaborations.
- Leading bullets with venue-first acceptance announcements.
- Choosing only low-star projects as flagships unless they are essential to the AMAP-specific spatial narrative.

## README Structure

Each flagship repository should use this order:

```md
# Project Name

One-sentence value proposition.

<teaser image or demo gif>

## Highlights
- What problem this solves.
- Why the method or benchmark is different.
- What users can reproduce or reuse.

## News
- YYYY.MM.DD Released code / model / data / benchmark.

## Quick Start
Install and run the smallest working demo.

## Main Results
Show the core benchmark table or qualitative comparison.

## Model, Data, and Demo
- Model:
- Dataset:
- Demo:
- Paper:

## Reproduce
Training, evaluation, and expected outputs.

## Citation

## Contact
```

## Minimum Release Bar

- A concise title and one-sentence value proposition.
- A teaser image, architecture figure, demo gif, or result visualization near the top.
- Installation instructions tested on a clean environment.
- A quick-start command that runs without reading the paper first.
- Links to paper, code, model, dataset, demo, or project page when available.
- Reproduction scripts for the main table or benchmark claim.
- License, citation, and contact information.
- Issue template or clear bug-report instructions.
- A short maintenance note if the repository is a paper artifact rather than a long-term project.

## Wording Pattern

Prefer contribution-first wording:

```md
MobilityBench evaluates route-planning agents in real-world mobility scenarios with scalable tasks, map-grounded constraints, and reproducible metrics.
```

Avoid venue-first wording as the main message:

```md
[ICLR 2026] Project Name
```

Venue information is still valuable, but it should support the contribution instead of replacing it.
