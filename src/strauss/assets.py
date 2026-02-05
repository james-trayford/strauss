# Use the pooch library as an asset manager
import pooch
from pathlib import Path

# URL for our STRAUSS assets
# TODO a more sensible domain!
ASSET_URL = "https://frontiers2024.icg.port.ac.uk/assets/"

# let's construct this manually for now, not feasible if we have a lot of stuff...
# TODO do this automatically from a repository?
asset_shas = {"glockenspiels.zip": "sha256:6af9719469816c7a13b944b5e3ff11e0358cae4881953a45f7c857b30adfcb0c",
              "mallets.zip": "sha256:ff6eaf530f567ed2cdc49398696284e1f1d3c7ba55b90ad7000e2526e4f96b86",
              "flute.sf2": "sha256:8efee668c98f548e14952970ed13234a4a6f21e39a1c6ad10dc8fd3fb0c642f2"}

# give them handy reference names
assets_names = {"glockenspiel": "glockenspiels.zip",
                "mallet": "mallets.zip",
                "flute": "flute.sf2"}

# create pooch instance
assets = pooch.create(
    path=pooch.os_cache("strauss"),
    base_url=ASSET_URL,
    registry=asset_shas
)

def get_asset_path(name):
    """
    Returns absolute path to the requested asset, download if doesn't exist..
    """

    filename = assets_names[name]
    
    if filename.endswith('.zip'):
        # do we need to unzip?        
        proc = pooch.Unzip
        file_path = assets.fetch(filename, processor=proc())
        return Path(file_path[0]).parents[0]
    else:
        file_path = assets.fetch(filename)
        return file_path
