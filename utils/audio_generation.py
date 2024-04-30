import pvorca


def generate_audio(text, filename, voice):
    orca = pvorca.create(
        access_key="89BlxJKCyiH/Eye4zhS74DxMibVpYlj/6qkLLw90NCm+ICw+AKYZqg==",
        model_path=voice,
    )
    orca.synthesize_to_file(text, filename)
