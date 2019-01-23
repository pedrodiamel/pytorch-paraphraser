import os
import requests
import tarfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download(namefile, id, destination='.' ):
    if not os.path.exists( destination ):
        print('Path {} not exits, we are create ..'.format(destination) )
        os.makedirs( destination )    
    print('Donwload file {}'.format(namefile) )
    download_file_from_google_drive(id,  os.path.join(destination, namefile ) )

def extract(pathname):
    # extract file
    cwd = os.getcwd()
    tar = tarfile.open( pathname, "r:gz")
    os.chdir(pathname)
    tar.extractall()
    tar.close()
    os.chdir(cwd)  


#download externs data
def download_extern_data( destination ):
    # destination = os.path.expanduser( '~/.datasets/txt' )
    id = '1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD'
    namefile = 'para-nmt-50m.zip'    
    download( namefile, id, destination )
    extract( os.path.join( destination, namefile ) )



# TODO January 23, 2019: Download pretrain models
#download models
def download_model():
    pass
   