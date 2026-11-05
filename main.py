"""Launch NightStack.  (Also the PyInstaller entry point.)"""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from nightstack.gui.app import main
    main()
