import os


def rename_files(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Sort files to ensure consistent ordering
    files.sort()

    # Initialize counter
    count = 1

    # Iterate through each file in the directory
    for filename in files:
        # Split the filename and extension
        name, ext = os.path.splitext(filename)

        # Create new filename with leading zeros
        new_filename = '{:03}{}'.format(count, ext)

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

        # Increment counter
        count += 1


# Replace 'directory_path' with the path to your directory
directory_path = 'clips'

rename_files(directory_path)
