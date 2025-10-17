# Speech MI F5-TTS

Some important things to note in this repository. <br/>

**dit.py** - contains the DiT hook code. <br/>

**cfm.py** - contains the CFM hook code. <br/>

**ablation_runner.py** - experiments and testing the patching hooks via model inference is done here. <br/>

**embedding_layer_calls.txt** - shows all submodules that can be called via our DiT hooks <br/>

**DiT Ablation Results** - Contains all the sounds files with the patching interventions done by ablation_runner.py. The test_explanation.txt describes each tested experiment. <br/>

**Attention Head Mechanism Check Folder** - Used Claude to modify the ablation_runner.py into using the hooks to extract the attention heads from F5-TTS. The results are in attention_statstics.json. But to summarize the data we can both extract and patch attention heads from F5-TTS attention block. <br/>

How to use our F5-TTS MI Code 

1. Create F5-TTS local editable library 
```
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -e .
```

Replace the DIT.py, CFM.py, and utils_infer.py with the code within this repo.

