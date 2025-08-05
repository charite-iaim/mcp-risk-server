
# README 

## Instructions

### Prerequisites
Create and activate virtual environment
```shell
python3 -m venv mcp-env
source mcp-env/bin/activate
```

Install missing packages
```shell
pip install -r requirements.txt
```

### Prepare your Data

### Configure Settings

Populate your configuration file, e.g., by duplicating `config.example.yaml` and naming it, e.g., to `my_config.yaml`:

```shell
cp config.example.yaml my_config.yaml
```
then edit the fields in `my_config.yaml` to select the risk score you want to compute on the case files and select the language model and give provider-specific details
```yaml
risk_score: HAS-BLED # or CHA2DS2-VASc, EuroSCORE II
provider: openai # supported: openai, deepseek, perplexity, qwen
model: gpt-4-1106-preview # or any other model name the provider gives you access to
```

Depending on which LLM provider you target, you have to set the following keys as environment variables to execute connection tests for your aimed provider:

| Provider | Environment Variable |
| -- | -- |
| Alibaba | `API_KEY` |
| DeepSeek | `API_KEY` |
| OpenAI | `API_KEY` (*sk-...*), `ORG_KEY` (*org-...*), `PROJECT_ID` (*proj_...*) |
| Perplexity | `API_KEY` | 

Advanced: You can set the keys permanently in a separate bash script `.bash_keys` with tight permissions. First create a separate file:
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

### Run Server
To run locally simply start the server in terminal via
```shell
python server.py
```
which will start the Scoring Server with the HTTP transport layer on port 8000. In terminal you should see output that indicates that the FastMCP server is running and is open for MCP communication on ` http://127.0.0.1:8000/mcp/`. You can quit the server anytime with Ctrl + C in terminal. 

### Run Client
As soon and as long the server is running, you can call the client simply by giving your config file:
```shell
python client.py my_config.yaml
```

### Output
Two basic output directories are created:

- Log dir: `./logs/<run_name>`
- Results dir: `./output/<run_name>`

LLM responses will be stored under `./logs/<run_name>/<case_id>` as one text file per query in the format `<item>_<timestamp>.log`. These files serve mainly for debugging.

Items extracted from the LLM-returned JSON string are aggregated over the cohort and stored in table `./outputs/<run_name>/intermediate/<score>/<score>_llm.csv`. Final risk scores calculated on `<score>_llm.csv` are placed into table `./outputs/<run_name>/final/<score>/<score>_calc.csv`.

## Add another Risk Score
To extend the MCP server app by a new risk score, two things have to be done:

- Define `NewRiskScore` as a derivative of the base class `RiskScore` and provide a `calculate` function
- Provide prompt instructions for information extraction under `prompts/newriskscore_template.yaml`
- optionally, add unit tests under `tests/scoring`

Finally, modify the configuration file to name the new risk score under `payload` -> `risk_score` and run the pipeline as described above.


### Unit Tests
Most unit tests do not require an LLM provider connection or use a mockup. 
However, the integration tests, a valid API key is needed.
To execute these tests, run from top level:
```shell
export PYTHONPATH=. # ensure local pytest is used
pytest tests  # Runs complete test suit
pytest tests/scoring  # Runs all scoring tests
pytest tests/pipeline  # Runs end-to-end pipeline tests

```
If you want or need to excempt end-to-end tests that require internet connection and a valid API key or vice versa, run either
```shell
pytest -m mock_llm  # Runs only tests marked as "mock_llm"
pytest -m llm_api   # Runs only tests marked as "llm_api"
```
Note, that only those API end-to-end tests are executed for which a key is found in the set of environment variables.


### Trouble Shooting

#### Terminate Server Process Not Working with CTRL + C
Kill server for restart or termination if ctrl + c does not work by identifying associated process id and kill
```shell
ps aux | grep server.py
kill <pid>
```

#### Repeated Warning That Items Could Not Be Extracted

Check the log outputs under `cfg['log_dir'] / score / case_id / <item>_timestamp.log` and verify that captured LLM output is matched by regex defined in the `Extractor` class in `provider_tools.py`
