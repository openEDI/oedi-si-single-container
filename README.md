# CLI

## Installation

```
pip install -e .
```

## Using the CLI

Before running the CLI make sure that the required Docker image is available. If not, the image can be built using,

```
oedisisc build --tag singlecontainerapp:0.3.0 -p python
```

if completely new build is required i.e. no previously cached container layers, use,

```
oedisisc build -t singlecontainerapp:0.3.0 --nocache true -p python
```

```
oedisisc init -p sampleProject
```

```
oedisisc run -p sampleProject -c sampleProject/config/user_config.json --tag singlecontainerapp:0.3.0
```

### Set/Get CLI default options

The default CLI settings can be set once using set_default option as shown below.

```
oedisisc set_default --tag singlecontainerapp:0.3.0 --python_cmd python --podman false
```

Since the default tag was set, the run command reduces to,

```
oedisisc run -p sampleProject -c sampleProject/config/user_config.json
```

instead of,

```
oedisisc run -p sampleProject -c sampleProject/config/user_config.json --tag singlecontainerapp:0.3.0
```

The view the default CLI settings use get_default option as shown below.

```
oedisisc get_default
```

### Using Podman instead of Docker

```
oedisisc build --tag singlecontainerapp:0.3.0 -p python --podman true
```

if completely new build is required i.e. no previously cached container layers, use,

```
oedisisc build -t singlecontainerapp:0.3.0 --nocache true -p python --podman true
```

```
oedisisc init -p sampleProject
```

```
oedisisc run -p sampleProject -c sampleProject/config/user_config.json --tag singlecontainerapp:0.3.0 --podman true
```


## Output

Check /path/to/project/output

## Integrating external/user federates without writing federate code

One can integrate DSSE (supported in this version) and DOPF (to be supported in the next version) in a manner that is completely decoupled from co-simulation and oedisi framework. Check /path/to/project/user_federates/user_dsse/main.py for more details.

# Deep Dive
* [Build and Runtime System](docs/build_and_runtime_system.md)
* [Data Preprocessor](docs/data_preprocessor.md)

# Edit Scenario
* [Prerequisite](docs/prerequisite.md)
* [Using CLI for Edit Scenario](docs/edit_scenario_uiruntime_uiserver.md)
* [Running Edit Scenario Directly Using Docker/Podman](docs/edit_scenario_uiruntime_uiserver_without_cli.md)
* [User Configuration Details](docs/user_config.md)
