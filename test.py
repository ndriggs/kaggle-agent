from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# result = api.competition_submit("submission.csv", message="Testing testing", competition="home-data-for-ml-course")

result = api.competition_download_files("home-data-for-ml-course", path=".", unzip=True) # NO KEYWORD UNZIP

print(result)

# competitions = api.competitions_list(search='housing prices')

# print(f'Found {len(competitions)} competitions matching "housing prices":')
# for comp in competitions:
#     print(f"- {comp}")