r"""Script to download INRIAPerson dataset from Kaggle

In order to use the Kaggle’s public API, you must first authenticate using an API token.
From the site header, click on your user profile picture, then on “My Account” from the dropdown menu.
This will take you to your account settings at https://www.kaggle.com/account.
Scroll down to the section of the page labelled API.

Dataset is stored in adversarial-person/data/
"""

!pip install kaggle

# Define username and public api key gathered from your Kaggle Account
KAGGLE_USERNAME = 'eulerismygod'
KAGGLE_PUBLIC_API_KEY = '81c12f2b229dae9c3469e25df4c04838'
api_token = {"username":KAGGLE_USERNAME,"key":KAGGLE_PUBLIC_API_KEY}

!mkdir ~/.kaggle

# Create .kaggle/kaggle.json file at the root for Kaggle API reference
with open('~/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

# Download dataset and extract it to adversarial-person/data
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d jcoral02/inriaperson

with zipfile.ZipFile('inriaperson.zip', 'r') as zip_ref:
    zip_ref.extractall('../data')

!rm inriaperson.zip
