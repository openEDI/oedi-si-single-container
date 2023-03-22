# Build

```
docker build -t singlecontainerapp:0.1.0 .
```

Or download the prebuilt image at "" and load it,

```
docker load -i singlecontainerapp.zip
```

# App structure

```
federates
deps
profiles
runner
logs
```
All the federate code is under *federates*. Dependencies for each federate are under *deps*. Each federate profile is under *profiles*. *runner* is the entrypoint and has config used for running the federates. Logs are stored in *logs* folder.

# What does this example demonstrate

Demonstrates how to source profiles to run different environments within the same container without issues. In this example, it is assumed that ANL DOPF uses cyipopt -- a python package which links to Ipopt shared library, which again links to linear solver shared libraries. PNNL DOPF also uses cyipopt within the same python environment i.e. no virtualenv (because that would not work for this case). However, PNNL builds bleeding edge version from the latest Ipopt and MUMPS code. The task then is to use profiles to make cyipopt (python rather) to connect to the latest version shared objects. The following is the dopf_pnnl_rc file which will be sourced.

```
export LD_LIBRARY_PATH=/home/app/deps/dopf_pnnl/lib:$LD_LIBRARY_PATH
```

Using the above profile, we will run,

```
#!/bin/bash
source /home/app/profiles/dopf_anl_rc && python3 /home/app/federates/mock_dopf_pnnl/main.py
```

You will notice that this is very similar to what we normally do. The only difference is that we now source the profile before running the app. That is it. The config used by the helics cli will now be,

```
"mock_dopf_pnnl": {
	"directory": "../federates/mock_dopf_pnnl",
	"name": "mock_dopf_pnnl",
	"exec": "/home/app/federates/mock_dopf_pnnl/run.sh",
	"hostname": "localhost"
}
```


# Running

## Single container with only public algorithms

### Prerequisites

The user can specify which algorithms to run using *user_config.json*. This file can be found in *runner/user_config.json*. For the rest of this document, it is assumed that the user has downloaded the entire repository. Hence, we will reference the configuration files under runner/*.json.

### Structure of user_config file

```
{
   "federates": [
      "mock_dopf_pnnl",
      "mock_dsse_pnnl",
      "mock_dopf_ornl",
      "mock_dsse_ornl",
      "mock_dopf_nrel",
      "mock_dsse_nrel",
      "mock_dopf_anl",
      "mock_dsse_anl",
      "sensor_power_real"
   ],
   "run_broker":true,
   "private_federates":[]
}
```
*federates* represent the algorithms that the user wants to run. The above are the available algorithms. Hence, once can also run the following, which will *only* run the listed federates.

```
{
   "federates": [
      "mock_dopf_pnnl",
      "sensor_power_real"
   ],
   "run_broker":true,
   "private_federates":[]
}
```

*run_broker* flag indicates whether the broker should be run in this container (more on this is discussed later).
*private_federates* are the federates that the user will run outside of this container, perhaps on host or on other container.

### Run

```
docker run -it --rm --name=gadalContainer -v $(pwd)/runner/user_config.json:/home/app/runner/user_config.json singlecontainerapp:0.1.0
```

## Single container with public and private algorithms

In this example, we will use *user_config_first_container.json* and *user_config_second_container.json*. In the first container, we will run a set of algorithms defined as,

```
{
   "federates": [
      "mock_dopf_pnnl",
      "mock_dsse_pnnl",
      "mock_dopf_ornl",
      "mock_dsse_ornl"
   ],
   "run_broker":true,
   "private_federates":[
      "mock_dopf_nrel",
      "mock_dsse_nrel",
      "mock_dopf_anl",
      "mock_dsse_anl"
   ]
}
```

The above instructs the first container runner to run the *federates* with a *broker* and tells the broker that additionally *private_federates* will join the cosim. The *private_federates* can then be run either on the host or on another container within the same machine.

```
{
   "federates": [
      "mock_dopf_nrel",
      "mock_dsse_nrel",
      "mock_dopf_anl",
      "mock_dsse_anl"
   ],
   "run_broker":false,
   "private_federates":[
      
   ]
}
```

The second container will run the remaining *federates*. Notice we do not list the set of federates for the second container under *private_federates* but rather provide that list under *federates*. The notion of *private_federates* in this context is simply federates that will be run outside of this run context but are expected to join the cosim. P.S. Although we run two containers we still use the same image *singlecontainerapp:0.1.0*.

```
docker run --rm --net=host --name=gadalContainer -v $(pwd)/runner/user_config_first_container.json:/home/app/runner/user_config.json -v $(pwd)/logs:/home/app/logs singlecontainerapp:0.1.0

docker run --rm --net=host --name=userContainer -v $(pwd)/runner/user_config_second_container.json:/home/app/runner/user_config.json -v $(pwd)/logs:/home/app/logs singlecontainerapp:0.1.0
```

From the root level folder on host we can check the logs, specifically the Ipopt version, using,

```
watch 'grep -ir "ipopt version" logs/'
```
