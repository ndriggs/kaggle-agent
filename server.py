from fastmcp import FastMCP
import pandas as pd
import os
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from tabpfn import TabPFNRegressor, TabPFNClassifier


# ------ code from https://github.com/arrismo/kaggle-mcp ------
def run_server():
    
    try:
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API Authenticated Successfully.")
    except Exception as e:
        print(f"Error authenticating Kaggle API: {e}")
        # api remains None if authentication fails

    # Initialize the FastMCP server
    mcp = FastMCP("kaggle-mcp")

    # @mcp.tool()
    # async def search_kaggle_datasets(query: str) -> str:
    #     """Searches for datasets on Kaggle matching the query using the Kaggle API."""
    #     if not api:
    #         # Return an informative error if API is not available
    #         return json.dumps({"error": "Kaggle API not authenticated or available."})

    #     print(f"Searching datasets for: {query}")
    #     try:
    #         search_results = api.dataset_list(search=query)
    #         if not search_results:
    #             return "No datasets found matching the query."

    #         # Format results as JSON string for the tool output
    #         results_list = [
    #             {
    #                 "ref": getattr(ds, 'ref', 'N/A'),
    #                 "title": getattr(ds, 'title', 'N/A'),
    #                 "subtitle": getattr(ds, 'subtitle', 'N/A'),
    #                 "download_count": getattr(ds, 'downloadCount', 0), # Adjusted attribute name
    #                 "last_updated": str(getattr(ds, 'lastUpdated', 'N/A')), # Adjusted attribute name
    #                 "usability_rating": getattr(ds, 'usabilityRating', 'N/A') # Adjusted attribute name
    #             }
    #             for ds in search_results[:10]  # Limit to 10 results
    #         ]
    #         return json.dumps(results_list, indent=2)
    #     except Exception as e:
    #         # Log the error potentially
    #         print(f"Error searching datasets for '{query}': {e}")
    #         # Return error information as part of the tool output
    #         return json.dumps({"error": f"Error processing search: {str(e)}"})


    # @mcp.tool()
    # async def download_kaggle_dataset(dataset_ref: str, download_path: str | None = None) -> str:
    #     """Downloads files for a specific Kaggle dataset.
    #     Args:
    #         dataset_ref: The reference of the dataset (e.g., 'username/dataset-slug').
    #         download_path: Optional. The path to download the files to. Defaults to '<project_root>/datasets/<dataset_slug>'.
    #     """
    #     if not api:
    #         # Return an informative error if API is not available
    #         return json.dumps({"error": "Kaggle API not authenticated or available."})

    #     print(f"Attempting to download dataset: {dataset_ref}")

    #     # Determine absolute download path based on script location
    #     # Use Path.cwd() if run via script entry point, or __file__ if run directly
    #     try:
    #         project_root = Path(__file__).parent.parent.resolve() # NEW: this is the parent of src/, i.e., the project root
    #     except NameError: # __file__ might not be defined when run via entry point
    #         project_root = Path.cwd() # NEW: Assume cwd is project root if __file__ is not defined


    #     if not download_path:
    #         try:
    #             dataset_slug = dataset_ref.split('/')[1]
    #         except IndexError:
    #             return f"Error: Invalid dataset_ref format '{dataset_ref}'. Expected 'username/dataset-slug'."
    #         # Construct absolute path relative to project root
    #         download_path_obj = project_root / "datasets" / dataset_slug # NEW
    #     else:
    #         # If a path is provided, resolve it relative to project root
    #         download_path_obj = project_root / Path(download_path) # NEW
    #         # Ensure it's fully resolved
    #         download_path_obj = download_path_obj.resolve()


    #     # Ensure download directory exists (using the Path object)
    #     try:
    #         download_path_obj.mkdir(parents=True, exist_ok=True)
    #         print(f"Ensured download directory exists: {download_path_obj}") # Will print absolute path
    #     except OSError as e:
    #         return f"Error creating download directory '{download_path_obj}': {e}"

    #     try:
    #         print(f"Calling api.dataset_download_files for {dataset_ref} to path {str(download_path_obj)}")
    #         # Pass the path as a string to the Kaggle API
    #         api.dataset_download_files(dataset_ref, path=str(download_path_obj), unzip=True, quiet=False)
    #         return f"Successfully downloaded and unzipped dataset '{dataset_ref}' to '{str(download_path_obj)}'." # Show absolute path
    #     except Exception as e:
    #         # Log the error potentially
    #         print(f"Error downloading dataset '{dataset_ref}': {e}")
    #         # Check for 404 Not Found
    #         if "404" in str(e):
    #             return f"Error: Dataset '{dataset_ref}' not found or access denied."
    #         # Check for other specific Kaggle errors if needed
    #         return f"Error downloading dataset '{dataset_ref}': {str(e)}"
    # ---------------- end of borrowed code -----------------------------------

    @mcp.tool("download_competition_data")
    def download_competition_data(competition_name: str, download_path: str = None):
        "Downloads competition data using the Kaggle API"
        if not api:
            return "Kaggle API is not authenticated."
        try:
            if not download_path:
                download_path = os.path.join(os.getcwd(), "competitions", competition_name)
            os.makedirs(download_path, exist_ok=True)
            result = api.competition_download_files(competition_name, path=download_path) # NOW WE HAVE TO UNZIP IT
            print("Competition download result: ", result)
            return result['message'] if 'message' in result else "Downloaded competition data to " + download_path
        except Exception as e:
            print(f"Error downloading competition data: {e}")
            return f"Error downloading competition data: {str(e)}"

    @mcp.tool("load_data")
    def load_data(train_file_path: str, test_file_path: str):
        df = pd.read_csv(train_file_path)
        mcp.session['train_df'] = df
        df_test = pd.read_csv(test_file_path)
        mcp.session['test_df'] = df_test
        print(f"Loaded train data from {train_file_path} with shape {df.shape}")
        return "Training data saved to train_df and test data saved to test_df"

    @mcp.tool("describe_data")
    def describe_data(df_name: str):
        "Returns summary statistics of the dataframe"
        print("Called describe_data for:", df_name)
        df = mcp.session[df_name]
        description = df.describe().to_dict()
        return description

    @mcp.tool("head_data")
    def head_data(df_name: str):
        "Returns the first few rows of the dataframe"
        print("Called head_data for:", df_name)
        df = mcp.session[df_name]
        return df.head().to_dict()

    @mcp.tool("value_counts")
    def value_counts(df_name: str, column_name: str):
        "Returns the value counts of a specific column in the dataframe"
        print("Called value_counts for:", df_name, column_name)
        df = mcp.session[df_name]
        counts = df[column_name].value_counts().to_dict()
        return counts

    @mcp.tool("drop_column")
    def drop_column(df_name: str, column_name: str, test_df_name: str = None):
        "Drops a column from the dataframe. If test_df_name is provided, drops the column from the test dataframe as well."
        print("Called drop_column for:", df_name, column_name)
        df = mcp.session[df_name]
        df = df.drop(columns=[column_name])
        mcp.session[df_name] = df
        if test_df_name:
            df_test = mcp.session[test_df_name]
            df_test = df_test.drop(columns=[column_name])
            mcp.session[test_df_name] = df_test
        return f"Dropped column {column_name} from {df_name}{' and ' + test_df_name if test_df_name else '.'}"

    @mcp.tool("create_is_na_column")
    def create_is_na_column(df_name: str, column_name: str, test_df_name: str = None):
        "Creates a new column indicating missing values in the specified column. If test_df_name is provided, creates the column in the test dataframe as well."
        print("Called create_is_na_column for:", df_name, column_name)
        df = mcp.session[df_name]
        df[f"{column_name}_is_na"] = df[column_name].isna()
        mcp.session[df_name] = df
        if test_df_name:
            df_test = mcp.session[test_df_name]
            df_test[f"{column_name}_is_na"] = df_test[column_name].isna()
            mcp.session[test_df_name] = df_test
        return f"Created is_na column for {column_name} in {df_name}{' and ' + test_df_name if test_df_name else '.'}"

    @mcp.tool("impute_missing_values")
    def impute_missing_values(df_name: str, column_name: str, strategy: str, test_df_name: str = None):
        "Imputes missing values in the specified column using the given strategy"
        "strategy can be 'mean', 'median', or 'mode'"
        "If test_df_name is provided, imputes missing values in the test dataframe as well."

        print("Called impute_missing_values for:", df_name, column_name, strategy)
        df = mcp.session[df_name]
        if strategy == "mean":
            df[column_name].fillna(df[column_name].mean(), inplace=True)
        elif strategy == "median":
            df[column_name].fillna(df[column_name].median(), inplace=True)
        elif strategy == "mode":
            df[column_name].fillna(df[column_name].mode()[0], inplace=True)
        mcp.session[df_name] = df
        if test_df_name:
            df_test = mcp.session[test_df_name]
            if strategy == "mean":
                df_test[column_name].fillna(df_test[column_name].mean(), inplace=True)
            elif strategy == "median":
                df_test[column_name].fillna(df_test[column_name].median(), inplace=True)
            elif strategy == "mode":
                df_test[column_name].fillna(df_test[column_name].mode()[0], inplace=True)
            mcp.session[test_df_name] = df_test
        return f"Imputed missing values in {column_name} of {df_name}{' and' + test_df_name if test_df_name else ''} using {strategy} strategy"

    @mcp.tool("fill_na")
    def fill_na(df_name: str, column_name: str, value, test_df_name: str = None):
        "Fills missing values in the specified column with a given value. If test_df_name is provided, fills missing values in the test dataframe as well."

        print("Called fill_na for:", df_name, column_name, value)
        df = mcp.session[df_name]
        df[column_name].fillna(value, inplace=True)
        mcp.session[df_name] = df
        if test_df_name:
            df_test = mcp.session[test_df_name]
            df_test[column_name].fillna(value, inplace=True)
            mcp.session[test_df_name] = df_test
        return f"Filled NA in {column_name} of {df_name}{' and' + test_df_name if test_df_name else ''} with {value}."

    @mcp.tool("copy_dataframe")
    def copy_dataframe(df_name: str, new_df_name: str):
        "Creates a copy of the dataframe"
        print("Called copy_dataframe for:", df_name, "as", new_df_name)
        df = mcp.session[df_name]
        mcp.session[new_df_name] = df.copy()
        return f"Created copy of {df_name} as {new_df_name}"

    @mcp.tool("list_dataframes")
    def list_dataframes():
        "Lists all dataframes in the session"
        print("Called list_dataframes")
        return list(mcp.session.keys())

    @mcp.tool("create_model")
    def create_model(model_name: str, model_type: str):
        "Creates a machine learning model"
        "Supported model types: RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, TabPFNRegressor, TabPFNClassifier"

        print("Called create_model for:", model_name, model_type)
        if model_type == "RandomForestRegressor":
            model = RandomForestRegressor()
        elif model_type == "RandomForestClassifier":
            model = RandomForestClassifier()
        elif model_type == "HistGradientBoostingRegressor":
            model = HistGradientBoostingRegressor()
        elif model_type == "HistGradientBoostingClassifier":
            model = HistGradientBoostingClassifier()
        elif model_type == "TabPFNRegressor":
            model = TabPFNRegressor()
        elif model_type == "TabPFNClassifier":
            model = TabPFNClassifier()
        else:
            return "Unsupported model type"

        mcp.session[model_name] = model
        return f"Created model {model_name} of type {model_type}"

    @mcp.tool("train_model")
    def train_model(model_name: str, df_name: str, target_column: str):
        "Trains a machine learning model"
        print("Called train_model for:", model_name, df_name, target_column)
        try : 
            model = mcp.session[model_name]
            df = mcp.session[df_name]
            
            X = df.drop(columns=[target_column])
            y = df[target_column]

            model.fit(X, y)
            mcp.session[model_name] = model
            return f"Trained model {model_name} on {df_name}"
        except Exception as e:
            return f"Error training model: {str(e)}"

    @mcp.tool("predict")
    def predict(model_name: str, df_name: str, id_col: str, target_col: str, path_to_save: str, predict_proba: bool = True):
        "Makes predictions using the trained model, saves to a csv ready for submission"
        try: 
            model = mcp.session[model_name]
            df = mcp.session[df_name]

            if predict_proba:
                predictions = model.predict_proba(df)
            else:
                predictions = model.predict(df)

            submission_df = pd.DataFrame({
                id_col: df[id_col],
                target_col: predictions
            })

            submission_df.to_csv(path_to_save, index=False)

            print("Predictions made and saved to:", path_to_save)
            return "Predictions saved to " + path_to_save
        
        except Exception as e:
            return f"Error making predictions: {str(e)}"


    @mcp.tool("submit_to_kaggle")
    def submit_to_kaggle(submission_file_path: str, competition_name: str, message: str = "Submission via MCP"):
        "Submits a predictions file to Kaggle"

        if not api:
            return "Kaggle API is not authenticated."

        try:
            result = api.competition_submit(submission_file_path, message=message, competition=competition_name)
            print("Kaggle submission result: ", result)
            return result['message'] if 'message' in result else "Submission result: " + str(result)
        except Exception as e:
            print(f"Error submitting to Kaggle: {e}")
            return f"Error submitting to Kaggle: {str(e)}"

    # ------ code from https://github.com/arrismo/kaggle-mcp ------
    print("Starting Kaggle MCP Server via mcp.run()...")

    # Call the run() method on the FastMCP instance
    # This likely contains the server startup logic used by the CLI
    mcp.run()

    # The code below this point will only execute after mcp.run() stops
    print("Kaggle MCP Server stopped.")

if __name__ == "__main__":
    run_server()