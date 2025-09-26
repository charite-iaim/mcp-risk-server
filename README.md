
# 🫀 FastMCP Risk Scoring Platform

## 🚀 Installation

### Prerequisites
Create and activate virtual environment
```shell
python3 -m venv .mcp-env
source .mcp-env/bin/activate
```

Install missing packages user mode:
```shell
pip install .
```

Optionally, install missing packages dev mode:
```shell
pip install -e .[dev]
```

---

## 📝 Data Preparation
Create a directory, e.g., named `data_dir` and place your patient reports as individual text files inside it.

**File Naming**: Each file should correspond to a single case; the case ID will be extracted from the file name by removing the extension. If an underscore (`_`) is present, only the part before the first underscore is used as the case ID.
**File Format**: plain text format and UTF-8 encoding recommended.

```shell
data_dir
      ├ Pa30df485.txt
      ├ P6d9b89a8.txt
            :
      └ P0b1d9044.txt
```

---

## ⚙️ Configuration

Populate your configuration file, e.g., by duplicating `config.example.yaml` and re-naming it to say `my_config.yaml`:

```shell
cp config.example.yaml my_config.yaml
```
then edit the fields in `my_config.yaml` to select the risk score you want to compute on the case files and select the language model and give provider-specific details
```yaml
risk_score: HAS-BLED # or CHA2DS2-VASc, EuroSCORE II
provider: openai # supported: openai, deepseek, perplexity, qwen
model: gpt-4-1106-preview # or any other model name the provider gives you access to
```

Depending on which LLM provider you selected, you have to set the following keys either as environment variables (option 1) or directly in the config (option 2):

api:
  api_key:      # Secret API key (e.g., sk-...)
  org_key:      # Organization key, if required (e.g., org-...)
  project_id:   # Project ID, if required (e.g., proj_...)


| Provider | Environment Variable | Config Variable |
| -- | -- | -- |
| Alibaba | `API_KEY` | `api:api_key` |
| DeepSeek | `API_KEY` | `api:api_key` |
| OpenAI | `API_KEY` (*sk-...*) | `api:api_key` |
| | `ORG_KEY` (*org-...*) | `api:org_key` |
| | `PROJECT_ID` (*proj_...*) | `api:project_id` |
| Perplexity | `API_KEY` | `api:api_key` |

**Advanced**: You can set the keys permanently in a separate bash script `.bash_keys` with tight permissions. First create a separate file:
```shell
vi ~/.bash_keys
```
and write needed keys into it, e.g., `API_KEY=<secret_api_key>`. Now, adjust rights such that file can only be read by current user:
```shell
chmod 400 ~/.bash_keys  
```
Allow automatic source execution upon `source ~/.bash_profile` call by inserting these three lines into you `~/.bash_profile`:
```shell
if [ -f ~/.bash_secrets ]; then
    source ~/.bash_secrets
fi
```

---

## 🖥️ Running the Server
After having configured the configuration file and prepared the text data, for a local run of the Scoring server simply type in repo root level 
```shell
python server.py
```
This will start the server with the HTTP transport layer on port 8000. In terminal you should see output that indicates that the FastMCP server is running and is open for MCP communication on ` http://127.0.0.1:8000/mcp/`. You can quit the server anytime with Ctrl + C in terminal. 

---

## 🤖 Client Usage
Assuming your server is up and running, you can call the client simply by giving your config file:
```shell
python client.py my_config.yaml
```
---

## 📊 Output
Two basic output directories are created, prefixed :

- Log dir: `./outputs/logs/<run_name>`
- Results dir: `./outputs/<run_name>`

Default folder is `outputs`, but can be changed under `outputs_dir` in your config.
For logging purposes LLM responses will be stored in the log dir separated by case id item-wise, in the format `<item>_<timestamp>.log`. `run_name` is taken from your config and `case_id`s extracted from text file prefixes as described here.


Important to you is that items extracted from the LLM-returned JSON strings are aggregated over all texts and stored in table `./outputs/<run_name>/stage1/<score>/<score>_llm.csv`. Final risk scores calculated on `<score>_llm.csv` are placed into table `./outputs/<run_name>/stage2/<score>/<score>_calc.csv`.

---

## 🧩 Customization: Adding a New Risk Score

To extend the MCP server app by a new risk score, two things have to be done:

- Define `NewRiskScore` as a derivative of the base class `RiskScore` and provide a `calculate` function
- Provide prompt instructions for information extraction under `prompts/newriskscore_template.yaml`
- optionally, add unit tests under `tests/scoring`

Finally, modify the configuration file to name the new risk score under `payload` -> `risk_score` and run the pipeline as described above.


## 🧪 Unit Testing
Most unit tests do not require an LLM provider connection or use a mockup. 

To execute these tests, run from top level:
```shell
export PYTHONPATH=. # ensure local pytest is used
pytest tests  # Runs complete test suit
pytest tests/scoring  # Runs all scoring tests
pytest tests/pipeline  # Runs end-to-end pipeline tests
```
To run integration tests with mocked LLM API calls (no API key required)
```shell
pytest -m mock_llm  # Runs only tests marked as "mock_llm"
```
Only tests decorated with `@pytest.mark.real_api` will require a valid API key set as **environment variable** and consume tokens from your account. In addition the provider and model name must be set. The tests will will run with the dummy data located under `.\tests\data\<score>`. By default the CHA2DS2-VASc score will be calculated from the reports under `.\tests\data\<score>` and compared to expected values of the fictitious patients whose true score is the value after underscore in their report names.

```shell
export TEST_PROVIDER=perplexity
export TEST_MODEL=sonar-small-online
export TEST_API_KEY=sk-...
export TEST_SCORE=cha2ds2vasc

pytest -m real_api   # Runs only tests marked as "real_api"
```

---

## 🛠️ Trouble Shooting

### Terminate Server Process Not Working with CTRL + C
Kill server for restart or termination if ctrl + c does not work by identifying associated process id and kill
```shell
ps aux | grep server.py
kill <pid>
```

### Repeated Warning That Items Could Not Be Extracted

Check the log outputs under `cfg['log_dir'] / score / case_id / <item>_timestamp.log` and verify that captured LLM output is matched by regex defined in the `Extractor` class in `provider_tools.py`
