# 🎬 ComfyUI Integration Guide: `qwen3-4b-deforum-prompt-lora`

## Technical Implementation Guide for "Deform" Film

> **Model:** `Limbicnation/qwen3-4b-deforum-prompt-lora`  
> **Base:** `Qwen/Qwen3-4B-Instruct-2507`  
> **Dataset:** `Limbicnation/deforum-prompt-lora-dataset` (5,400 examples)  
> **Target:** ComfyUI "Prompt Generator" Node

---

## 1. Node Setup Guide

### Parameter Checklist for "Prompt Generator" Node

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model Path** | `Limbicnation/qwen3-4b-deforum-prompt-lora` | HuggingFace Hub path |
| **Base Model** | `Qwen/Qwen3-4B-Instruct-2507` | Required for PEFT merging |
| **LoRA Name** | `qwen3-4b-deforum-prompt-lora` | Adapter identifier |
| **LoRA Strength** | `0.85 - 1.0` | Higher = stronger De Forum aesthetic |
| **Max Tokens** | `512` | Matches training config |
| **Context Window** | `1024` | Full sequence length |

### ComfyUI Node Configuration (JSON)

```json
{
  "class_type": "PromptGenerator",
  "inputs": {
    "model_path": "Limbicnation/qwen3-4b-deforum-prompt-lora",
    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
    "lora_name": "qwen3-4b-deforum-prompt-lora",
    "strength": 0.9,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "seed": -1,
    "device": "cuda"
  }
}
```

---

## 2. Syntax Protocol

### Trigger Words & Formatting Rules

**Primary Trigger Tokens:**
- `De Forum` - Activates the art film aesthetic
- `cinematic art film style` - Core style identifier
- `Scene {N}` - Scene reference for narrative continuity
- `noir-influenced` - Lighting/mood directive

**Input Format Template:**
```
### Instruction:
Generate a cinematic video prompt for:
Scene: {SCENE_NUMBER}
Context: {SCENE_DESCRIPTION}
Camera Movement: {CAMERA_TYPE}
Style: De Forum Art Film aesthetic

### Response:
```

**Required Output Structure:**
```
Cinematic {style} style video. 
scene: {Scene Name}. 
description: {Scene description}. 
camera movement: {Movement type}. 
lighting: chiaroscuro, moody atmospheric lighting with dramatic shadows. 
film style: grain texture, anamorphic lens characteristics, shallow depth of field. 
mood: contemplative, mysterious, emotionally resonant, noir-influenced
```

### Tag Activation Keywords

| Tag Category | Trigger Words |
|--------------|---------------|
| **Visual Style** | `noir`, `minimalist`, `chiaroscuro`, `anamorphic` |
| **Mood** | `contemplative`, `mysterious`, `emotional`, `atmospheric` |
| **Genre** | `art film`, `tech-noir`, `existential`, `humanist` |
| **Technical** | `grain texture`, `shallow depth of field`, `film grain` |

---

## 3. Scene 17 & 18 Examples

### Scene 17: "The Revelation"

**Narrative Context:**
Scene 17 represents the climactic revelation moment in the De Forum narrative arc. Sarah discovers the truth about the anonymous sender and confronts the nature of her artistic block.

**Optimized Input String:**
```text
### Instruction:
Generate a cinematic video prompt for:
Scene: Scene 17
Context: INT. ABANDONED THEATER - NIGHT. Sarah stands center stage, illuminated by a single spotlight. The vast empty auditorium stretches into darkness around her. In her hands, she holds the final piece of the puzzle—an antique film reel that contains the truth she's been seeking. Her expression transforms from fear to understanding.
Camera Movement: slow dolly back revealing vast empty space
Style: De Forum Art Film aesthetic, climactic revelation, psychological intensity

### Response:
```

**Expected Output Format:**
```text
Cinematic cinematic art film style video. scene: Scene 17. description: INT. ABANDONED THEATER - NIGHT. Sarah stands center stage, illuminated by a single spotlight. The vast empty auditorium stretches into darkness around her. In her hands, she holds the final piece of the puzzle—an antique film reel that contains the truth she's been seeking. Her expression transforms from fear to understanding, a moment of profound revelation captured in stark isolation. camera movement: slow dolly back revealing vast empty space. lighting: chiaroscuro, single spotlight creating dramatic isolation, deep shadows consuming the frame. film style: grain texture, anamorphic lens characteristics, shallow depth of field, high contrast black and white aesthetic. mood: revelatory, intense, psychologically charged, noir-influenced, transcendent
```

**Technical Parameters for Scene 17:**
```json
{
  "aspect_ratio": "2.39:1",
  "gen_model": "WanVideo",
  "seed": 174017,
  "guidance_scale": 8.0,
  "steps": 50,
  "tags": ["noir", "revelation", "theater", "climactic", "existential", "atmospheric"]
}
```

---

### Scene 18: "The Transformation"

**Narrative Context:**
Scene 18 depicts the aftermath and transformation—Sarah emerges changed, integrating her experience into a new artistic vision. The final scene provides narrative closure while maintaining open interpretation.

**Optimized Input String:**
```text
### Instruction:
Generate a cinematic video prompt for:
Scene: Scene 18
Context: EXT. CITY ROOFTOP - DAWN. Sarah stands at the edge of the rooftop, facing the emerging sunrise. The city skyline stretches before her, bathed in golden hour light. She holds her camera, ready to capture the world anew. The weight of her past struggles has lifted, replaced by quiet determination.
Camera Movement: slow crane shot ascending to reveal cityscape
Style: De Forum Art Film aesthetic, transcendence, quiet resolution, new beginning

### Response:
```

**Expected Output Format:**
```text
Cinematic cinematic art film style video. scene: Scene 18. description: EXT. CITY ROOFTOP - DAWN. Sarah stands at the edge of the rooftop, facing the emerging sunrise. The city skyline stretches before her, bathed in golden hour light. She holds her camera, ready to capture the world anew. The weight of her past struggles has lifted, replaced by quiet determination and artistic rebirth. camera movement: slow crane shot ascending to reveal cityscape. lighting: chiaroscuro transitioning to soft dawn light, moody atmospheric lighting with dramatic shadows giving way to warm golden tones. film style: grain texture, anamorphic lens characteristics, shallow depth of field, ethereal morning haze. mood: transcendent, hopeful, quietly resolved, emotionally resonant, noir-influenced
```

**Technical Parameters for Scene 18:**
```json
{
  "aspect_ratio": "2.39:1",
  "gen_model": "CogVideoX",
  "seed": 180018,
  "guidance_scale": 7.5,
  "steps": 50,
  "tags": ["transcendence", "dawn", "cityscape", "resolution", "atmospheric", "humanist"]
}
```

---

## 4. Inference Settings

### Recommended Generation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Temperature** | `0.65 - 0.75` | Lower for consistency, higher for creative variation |
| **Top-p** | `0.75 - 0.85` | Nucleus sampling for coherent outputs |
| **Top-k** | `20` | Limits vocabulary to most likely tokens |
| **Repetition Penalty** | `1.1` | Prevents repetitive phrases |
| **Length Penalty** | `1.0` | Neutral length preference |

### Optimal Configuration by Use Case

**High Consistency Mode** (Production):
```json
{
  "temperature": 0.65,
  "top_p": 0.75,
  "top_k": 20,
  "repetition_penalty": 1.15,
  "do_sample": true
}
```

**Creative Variation Mode** (Exploration):
```json
{
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "repetition_penalty": 1.05,
  "do_sample": true
}
```

**Balanced Mode** (Default):
```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "repetition_penalty": 1.1,
  "do_sample": true
}
```

---

## 5. ComfyUI Node Implementation Code

```python
# Custom node implementation reference for prompt_generator_node.py

class PromptGenerator:
    """ComfyUI custom node for qwen3-4b-deforum-prompt-lora inference."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_number": ("INT", {"default": 1, "min": 1, "max": 100}),
                "scene_description": ("STRING", {"multiline": True, "default": ""}),
                "camera_movement": (["slow tracking shot following subject",
                                     "static wide shot with subtle zoom",
                                     "handheld camera with slight shake",
                                     "360-degree orbit around subject",
                                     "slow push-in on face",
                                     "overhead crane shot descending"],),
                "mood_tags": ("STRING", {"default": "noir, atmospheric, contemplative"}),
            },
            "optional": {
                "lora_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")  # prompt, negative_prompt, tags
    RETURN_NAMES = ("video_prompt", "negative_prompt", "tags")
    FUNCTION = "generate_prompt"
    CATEGORY = "prompt_generation/deforum"
    
    def generate_prompt(self, scene_number, scene_description, camera_movement, 
                       mood_tags, lora_strength=0.9, temperature=0.7, max_tokens=512):
        """Generate video diffusion prompt using fine-tuned LoRA."""
        
        # Construct instruction following training format
        instruction = f"""### Instruction:
Generate a cinematic video prompt for:
Scene: Scene {scene_number}
Context: {scene_description}
Camera Movement: {camera_movement}
Style: De Forum Art Film aesthetic

### Response:"""
        
        # Generate using PEFT model
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract components (parse response format)
        video_prompt = self._extract_prompt(response)
        negative_prompt = self._get_negative_prompt()
        tags = mood_tags
        
        return (video_prompt, negative_prompt, tags)
    
    def _get_negative_prompt(self):
        """Return standard De Forum negative prompt."""
        return ("muted, bad eyes, dull, hazy, muddy colors, blurry, mutated, "
                "deformed, noise, stock image, borders, frame, watermark, text, "
                "signature, username, cropped, out of frame, bad composition, "
                "poorly rendered face, poorly drawn hands, low resolution, "
                "cartoonish style, futuristic elements, bright colors")
```

---

## 6. Reference Alignment

### De Forum Aesthetic Constraints

**Visual Language:**
- ✅ Chiaroscuro lighting with dramatic shadows
- ✅ Film grain and anamorphic lens characteristics
- ✅ Shallow depth of field
- ✅ Noir-influenced color palette
- ✅ Minimalist composition

**Narrative Elements:**
- ✅ Psychological depth
- ✅ Urban atmosphere
- ✅ Emotional resonance
- ✅ Contemplative mood
- ✅ Humanist themes

**Technical Specifications:**
- ✅ Aspect ratios: 16:9, 2.39:1, 1.85:1, 4:3
- ✅ Guidance scale: 7.0 - 8.5
- ✅ Compatible models: WanVideo, CogVideoX, AnimateDiff, SVD

---

## 7. Quick Reference Card

### Camera Movement Options
- `slow tracking shot following subject`
- `static wide shot with subtle zoom`
- `handheld camera with slight shake`
- `low angle dolly forward`
- `overhead crane shot descending`
- `360-degree orbit around subject`
- `slow push-in on face`
- `pan across landscape`
- `tilt up from ground level`
- `whip pan transition`
- `slow zoom out revealing context`
- `steadicam following movement`

### De Forum Tags
`noir`, `minimalist`, `urban`, `psychological`, `cinematic`, `art film`, `surreal`, `dystopian`, `tech-noir`, `emotional`, `dramatic`, `moody`, `atmospheric`, `contemplative`, `mysterious`, `dark`, `neo-noir`, `existential`, `philosophical`, `humanist`

### Aspect Ratios
- `16:9` - Standard widescreen
- `2.39:1` - Cinematic anamorphic
- `1.85:1` - Academy flat
- `4:3` - Classic/retro

---

## 8. Training Configuration Reference

```yaml
# LoRA Configuration
lora_r: 32
lora_alpha: 64
lora_dropout: 0.1
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Quantization
use_4bit: true
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_quant_type: "nf4"

# Training
learning_rate: 5.0e-5
num_train_epochs: 3
max_seq_length: 1024
```

---

## 9. Quick Prompt Examples for ComfyUI Node

Short prompts ready to paste into the Prompt Generator node input field. The LoRA fills in cinematic details automatically.

### Scene-specific

```
Scene 5: empty studio, scattered art supplies, slow push-in. De Forum contemplative.
```

```
Scene 12: rain-slicked alley, Sarah running, handheld camera. De Forum tech-noir.
```

```
Scene 15: Sarah logs back into De Forum, confrontation, static wide shot. De Forum psychological.
```

```
Scene 17: abandoned theater, revelation moment, single spotlight, dolly back. De Forum noir.
```

```
Scene 18: rooftop at dawn, artistic rebirth, crane shot ascending. De Forum transcendence.
```

### Mood-driven (no scene number)

```
Dark corridor, flickering fluorescent lights, handheld with slight shake. De Forum mysterious.
```

```
City bridge at twilight, lone figure, rain reflections, slow tracking shot. De Forum atmospheric.
```

```
Empty gallery, abstract paintings, overhead crane descending. De Forum minimalist, contemplative.
```

```
Underground club, strobe lighting, whip pan transition. De Forum surreal, tech-noir.
```

```
Foggy waterfront, distant sirens, static wide shot with subtle zoom. De Forum noir, existential.
```

---

*Generated for the De Forum Art Film "Deform" project.*
*Last updated: 2026-02-15*
