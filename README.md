# CLI

## Installation

```
pip install -e .
```

## Using the CLI

Before running the CLI make sure that the required Docker image is available. If not, the image can be built using,

```
oedisisc build --tag singlecontainerapp:0.2.1
```

if completely new build is required i.e. no previously cached container layers, use,

```
oedisisc build -t singlecontainerapp:0.2.1 --nocache true
```

```
oedisisc init -p sampleProject
```

```
oedisisc run -p sampleProject -c sampleProject/config/user_config.json --tag singlecontainerapp:0.2.1
```

## Output

Check /path/to/project/output

## Integrating external/user federates without writing federate code

One can integrate DSSE (supported in this version) and DOPF (to be supported in the next version) in a manner that is completely decoupled from co-simulation and oedisi framework. Check /path/to/project/user_federates/user_dsse/main.py for more details.

# Build and Runtime System

## Introduction
The build system allows combining the dockerfiles for each application and builds a single image.

* The *oedisi_runtime* consists of the feeder, sensor and recorder federates.
	* The relevant code for *oedisi_runtime* comes from sgidal-examples and oedisi repositories.
	* The singlecontainerapp will maintain the relevant updates.
* In order to include DSSE and DOPF federates, a new folder under build/ should be created. For example, build/pnnl_dsse. The contents of this folder should be as follows,
	* A *Dockerfile* that contains only the definition for the federate. See build/psse_dsse/Dockerfile for example. This is a mandatory file.
	* Optional *copy_statements.txt* that contains all the *COPY* commands that should be a part of the Dockerfile. See build/oedisi/copy_statements.txt for example. This implies that the *Dockerfile* should not contain any *COPY* statements. This allows better caching of layers which allows for faster build time.
	* Files that are to be copied.

* The runtime system allows configuring the co-simulation.

## Build System
cd to the root directory of this repository and then run the following command,

```
python3 build.py -t singlecontainerapp:0.2.1
```

## Runtime System

### Configuration Options

* **simulation_config** -- Allows specifying the following co-simulation options.

		"simulation_config": {
			"opendss_location": "/home/data/system/gadal_ieee123/qsts",
			"profile_location": "/home/data/system/gadal_ieee123/profiles",
			"sensor_location": "/home/data/system/gadal_ieee123/sensors.json",
			"start_date": "2017-01-01 00:00:00",
			"number_of_timesteps": 4,
			"run_freq_sec": 900
		}

* **use_oedisi_runtime** -- When set to **true** the oedisi runtime components (feeder, sensors, recorders) will be made part of the co-simulation.

* **oedisi_runtime_federates** -- Specifies the use of DSSE and DOPF federates that are shipped with OEDISI. The following will instruct the runtime system to use nrel and pnnl DSSE federates.

		"oedisi_runtime_federates": [
			"state_estimator_nrel",
			"state_estimator_pnnl"
		]


* **run_broker** -- When set to **true** will run the HELICS broker in this container.

* **user_provided_federates** -- Will run user provided federates.

		"user_provided_federates": []

* **externally_connected_federates** -- Allows external connection of federates that are part of the co-simulation. Here, external is relative to the container. The externally connected federates can be from the host, or other container(s).

Putting things together,

```
{
   "name": "runtime_example",
   "use_oedisi_runtime": true,
   "oedisi_runtime_federates": [
      "state_estimator_nrel",
      "state_estimator_pnnl"
   ],
   "run_broker": true,
   "user_provided_federates": [],
   "externally_connected_federates": [],
   "simulation_config": {
      "opendss_location": "/home/data/system/gadal_ieee123/qsts",
      "profile_location": "/home/data/system/gadal_ieee123/profiles",
      "sensor_location": "/home/data/system/gadal_ieee123/sensors.json",
      "start_date": "2017-01-01 00:00:00",
      "number_of_timesteps": 4,
      "run_freq_sec": 900
   }
}
```

### Running the co-simulation with no external federates
cd to the root directory of this repository and then run the following command. Replace *-v $(pwd)/runner/user_config.json* with path to the configuration file you want to use and $(pwd)/output with the path to the output files.

```
docker run --rm -it -v $(pwd):/home/runtime -v $(pwd)/runner/user_config.json:/home/runtime/runner/user_config.json -v $(pwd)/output:/home/output singlecontainerapp:0.2.1
```
or on windows,

```
docker run --rm -it -v %cd%:/home/runtime -v %cd%/runner/user_config.json:/home/runtime/runner/user_config.json -v %cd%/output:/home/output singlecontainerapp:0.2.1
```
