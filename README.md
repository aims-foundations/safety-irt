Download FINALMERGEDTAGGED.csv and anchors.csv
https://huggingface.co/datasets/MaxZ119/safetyirt/tree/main

clone Github, add the two csv files to "model" folder.
chmod +x reproduce.sh
./reproduce.sh


# Decoupling Safety Alignment from Translation Difficulty: A Multi-Group IRT Approach

### 1. Motivation & Problem Formulation
Large Language Models (LLMs) show significant safety degradation in non-English, low-resource languages like Swahili and Javanese. Current metrics like Jailbreak Success Rate (JSR) use binary Safe/Unsafe labels, which fail to distinguish between a model's lack of safety alignment and the inherent difficulty introduced by translation.

This project utilizes a **Multi-Group Item Response Theory (IRT)** framework to decouple these factors, allowing for more targeted alignment and fairer benchmarking.

### 2. Theoretical Framework
We use a **Many-Facet Rasch Model** to jointly estimate safety parameters. The probability of a safe response is modeled as:

P(Safe) = sigma(Model_Ability - (Prompt_Difficulty + Language_Fluency_Shift + Translation_Safety_Cost)) 

#### Key Parameters:
**Base Safety Capability (theta):** The language-agnostic safety "ability" of the model.

**Base Difficulty (beta):** The intrinsic difficulty of the prompt, derived from English.
**Fluency Shift (gamma):** The global difficulty increase of processing a specific language.
**Translation Safety Cost (tau):** The prompt-specific drift representing how much translation distorts the safety concept.

We apply a **hierarchical shrinkage prior** (e.g., Horseshoe) to the translation cost terms to ensure stability and mitigate confounding factors.

### 3. Methodology
**Dataset:** We utilize the **MultiJail** dataset, featuring 3,150 prompts across 10 languages and 18 safety categories.
**Models:** Evaluation includes 20 open-source models from the **Llama 3**, **Gemma 2**, and **Qwen 2.5** families.
**Grading:** We employ **LLM-as-a-Judge** (GPT-4o) to grade ~60,000 responses, with a validation panel consisting of Claude 3.5 Sonnet and Qwen to mitigate bias.
**Implementation:** Parameters are estimated via Variational Inference using the **py-irt** library.

### 4. Ideal Results & Hypotheses
**The Cost is Real:** We expect the Translation Safety Cost to be positive and statistically significant for low-resource languages.
**Rank Reversal:** We anticipate identifying models that are unfairly penalized by JSR but actually possess high intrinsic safety capabilities.
**Concept Alignment:** We hypothesize that physical harm categories (e.g., weapons) will have higher translation costs than abstract ones.
