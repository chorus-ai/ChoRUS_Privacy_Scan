
# Running the Privacy Scan Tool with Docker

This document explains how to run the Privacy Scan Tool using Docker, including details on setting up the configuration file `config.json` and specifying the necessary directories for volumes.

## Command to Run the Docker Container

To run the Privacy Scan Tool Docker container, use the following command:

```bash
docker run -v $(pwd)/output:/privacy_scan_tool/output -v $(pwd)/config.json:/privacy_scan_tool/config.json -v $(pwd)/data_folder:/privacy_scan_tool/data_folder luyaochen/privacy_scan_tool:20240517
```

### Explanation of the Docker Command

- `docker run`: Runs the Docker container.
- `-v $(pwd)/output:/privacy_scan_tool/output`: Maps the local `output` directory to the container's `/privacy_scan_tool/output` directory.
- `-v $(pwd)/config.json:/privacy_scan_tool/config.json`: Maps the local `config.json` file to the container's `/privacy_scan_tool/config.json` file.
- `-v $(pwd)/data_folder:/privacy_scan_tool/data_folder`: Maps the local `data_folder` directory to the container's `/privacy_scan_tool/data_folder` directory.
- `luyaochen/privacy_scan_tool:20240517`: Specifies the Docker image to use.

## Creating Required Local Directories

Before running the Docker command, ensure you create the necessary local directories. Run the following commands to create them:

```bash
mkdir -p output
mkdir -p data_folder
```

## Configuring `config.json`

The `config.json` file contains various settings for the Privacy Scan Tool. Below is an example configuration file:

```json
{
    "available_dbs": {
        "PSQL_MIMIC": ["postgresql://userid:password@192.168.0.100:5432/mimic", "mimiciii"],
        "LOCAL_TEXT_FILES": "LOCAL_TEXT_FILES"
    },
    "text_file_location": "./data_folder", 
    "output_folder": "./output",
    "result_file": "phi_scan_results.xls",

    "selected_db": "PSQL_MIMIC",
    "tables_to_scan" : ["admissions","patients"],

    "data_profile_sample_size": 1000,
    "PHI_SCAN_MODEL": "./phi_scan/XGBClassifier(V220240514).json"
}
```

### Configuration Details

- **available_dbs**: Lists the available databases. 
  - `PSQL_MIMIC`: PostgreSQL database connection details and the database name.
  - `LOCAL_TEXT_FILES`: Placeholder for using local text files.
- **text_file_location**: Directory location for local text files.
- **output_folder**: Directory for the output results.
- **result_file**: Name of the result file.
- **selected_db**: Specifies the database to use (`PSQL_MIMIC` or `LOCAL_TEXT_FILES`).
- **tables_to_scan**: Lists the tables or files to scan.
- **data_profile_sample_size**: Specifies the sample size for data profiling.
- **PHI_SCAN_MODEL**: Path to the PHI scan model.

### Configurations for Different Scenarios

#### Using Local Text Files

If you want to use local text files, modify the `config.json` as follows:

```json
{
    "available_dbs": {
        "PSQL_MIMIC": ["postgresql://userid:password@192.168.0.100:5432/mimic", "mimiciii"],
        "LOCAL_TEXT_FILES": "LOCAL_TEXT_FILES"
    },
    "text_file_location": "./data_folder", 
    "output_folder": "./output",
    "result_file": "phi_scan_results.xls",

    "selected_db": "LOCAL_TEXT_FILES",
    "tables_to_scan" : ["noshow.csv"],

    "data_profile_sample_size": 1000,
    "PHI_SCAN_MODEL": "./phi_scan/XGBClassifier(V220240514).json"
}
```

#### Using PostgreSQL Database

If you want to use a PostgreSQL database, ensure the `config.json` is set as follows:

```json
{
    "available_dbs": {
        "PSQL_MIMIC": ["postgresql://userid:password@192.168.0.100:5432/mimic", "mimiciii"],
        "LOCAL_TEXT_FILES": "LOCAL_TEXT_FILES"
    },
    "text_file_location": "./data_folder", 
    "output_folder": "./output",
    "result_file": "phi_scan_results.xls",

    "selected_db": "PSQL_MIMIC",
    "tables_to_scan" : ["person","observation"],

    "data_profile_sample_size": 1000,
    "PHI_SCAN_MODEL": "./phi_scan/XGBClassifier(V220240514).json"
}
```

Ensure you replace `postgresql://userid:password@192.168.0.100:5432/mimic` with your actual PostgreSQL connection details and the database name.

## Running the Tool

1. Ensure Docker is installed and running on your machine.
2. Create the necessary local directories by running:
    ```bash
    mkdir -p output
    mkdir -p data_folder
    ```
3. Place the `config.json` file in the current working directory.
4. Place any required data files in the `data_folder` directory.
5. Execute the Docker run command provided above.
6. The tool will process the data according to the configuration and output the results to the specified `output` directory.

By following these steps, you can successfully run the Privacy Scan Tool using Docker with the appropriate configuration.
