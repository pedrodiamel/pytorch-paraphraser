from __future__ import unicode_literals


def main():

    import phrases

    print()
    print('+---------------------------+')
    print('| Welcome to phrases :D !!! |')
    print('+---------------------------+')
    print('Version: ', phrases.__version__ )
    print()

    from phrases.webapp import app, net

    net.run(pathconfigurate='modelsconfig.json' )
    app.run(host='0.0.0.0', debug=False, port=3500)


if __name__ == '__main__':
    main()