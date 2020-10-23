from pywinauto.application import Application
from os import walk, path

# settings
path_from = 'G:/Dev/DataScience/dfma-map-file-archive/maps'
path_to   = 'G:/Dev/DataScience/masterthesis_gan_mapdesign/data/Dwarf Fortress Maps/PNG Exports'

# error list and file
errors = []
with open('errors.txt', 'r') as fobj:
    for line in fobj:
        # remove 'newline' character to enable comparison later on
        errors.append(line.replace('\n',''))


def get_files():
    # get list of files to decode
    filelist = []
    for root, dirs, files in walk(path_from):
        for file in files:
            if not path.exists(path.join(path_to.replace('/','\\'), file.replace('.fdf-map','.zip').replace('.FDF-MAP','.zip'))) \
            and not path.exists(path.join(path_to.replace('/','\\'), file.replace('.fdf-map','.png').replace('.FDF-MAP','.png'))) \
            and path.join(root, file.replace('\\\\','\\')) not in errors:
                # add to list of files to be processed if not yet available in output path in order to enable process restarts
                filelist.append(path.join(root, file))
            else:
                pass

    print(f'found {len(filelist)} map files to decode.')
    return filelist

def open_application():
    # open application
    app = Application(backend='win32').start('F:/Programs/DFMap Compressor/DwarfFortressMapCompressor.3.3.4.exe')
    # attach to application window
    dlg = app.window(title='Dwarf Fortress Map Compressor - version 3.3.4')
    dlg.wait('visible')

    # get list of available identifiers
    # dlg.print_control_identifiers()

    # switch to 'advanced mode'
    dlg['Switch to Advanced Interface'].click()

    return dlg, app

def decode(filelist, dlg, app):
    # iterate over all files in source path
    try:
        for i, source_file in enumerate(filelist, start=1):

            print(f'Currently working on file number {i} - {source_file}')
            # attach to new sub-windows and cl
            dlg = app.window(title='Dwarf Fortress Map Compressor - version 3.3.4')
            dlg.wait('visible', timeout=60)  # timeout needs to be set to about 1 minute due to large maps that take a while to decode

            # dlg.print_control_identifiers()
            # Click 'Decode' button
            dlg['&Decode: .df-map to .png\r\n(Can also decode .fdf-map)'].click()

            # attach to 'open file dialog'
            dlg = app.window(title='Select the encoded map')
            dlg.wait('visible') 
            #dlg.print_control_identifiers()

            # enter next source file
            dlg.type_keys(source_file.replace('/','\\').replace('(','{(}').replace(')','{)}'), with_spaces = True, )    # 'special' characters such as paranthesis need to be escaped with {} for type_key() to apply correctly
            dlg['&Open'].click()

            # attach to 'save in dialog'
            dlg = app.window(best_match='Where do you want to save the fortress map image?') # depending on whether there are 1 or more resulting PNGs, the window title changes. With the 'best_match' attribute we can still find the right window
            dlg.wait('visible')
            #dlg.print_control_identifiers()

            # enter target path
            dlg.type_keys(path_to.replace('/','\\'), with_spaces = True)
            dlg['&Save'].click()
            # click 'Save' again to confirm the file name in the new location
            dlg['&Save'].click()

            # attach back to main window
            #dlg = app.window(title='Dwarf Fortress Map Compressor - version 3.3.4')
            #dlg.wait('visible')

    except:
        print(f'time out error with file {source_file}')
        # add errored-out file to list and persist errors between runs
        with open('errors.txt', 'a') as fobj:
            fobj.write(source_file + '\n')

        # attach to error message windows
        dlg = app.top_window()
        # click 'Quit' button
        dlg['&Quit'].click()

        # restart the application
        dlg, app = open_application()
        # get new list of files (without the one the errored out and all previous entries)
        filelist = filelist[i:]
        # restart decode process
        decode(filelist, dlg, app)

def main():
    filelist = get_files()
    dlg, app = open_application()
    decode(filelist, dlg, app)

if __name__ == '__main__':
    main()

