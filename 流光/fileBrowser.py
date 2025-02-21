event = st_file_browser(
    def sort(files):
        return sorted(files, key=lambda x: x["size"])
    
    event = st_file_browser(
        os.path.join(current_path, "..", "example_artifacts"),
        file_ignores=("a.py", "a.txt", re.compile(".*.pdb")),
        key="A",
        show_choose_file=True,
        show_choose_folder=True,
        show_delete_file=True,
        show_download_file=True,
        show_new_folder=True,
        show_upload_file=True,
        show_rename_file=True,
        show_rename_folder=True,
        use_cache=True,
        sort=sort,
    )
