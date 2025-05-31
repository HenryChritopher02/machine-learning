#!/usr/bin/perl
use strict;
use warnings;

# --- Get the configuration file path from the first command-line argument ---
my $config_file_path = $ARGV[0];

if (not defined $config_file_path) {
    die "Usage: $0 <path_to_config_file.txt>\nLigand PDBQT file paths should be piped via STDIN.\n";
}
if (not -f $config_file_path) {
    die "Error: Config file '$config_file_path' not found or is not a regular file.\n";
}
print "Using Vina config file: $config_file_path\n";
# --- End of config file handling ---

print "Reading ligand PDBQT file paths from STDIN...\n";
# The original script read a filename from STDIN, then read that file.
# If your Python script pipes the *actual paths*, one per line, directly to STDIN,
# then we just read STDIN directly into @arr_file.
my @arr_file_paths = <STDIN>; # Reads all lines from STDIN

if (not @arr_file_paths) {
    die "Error: No ligand file paths received via STDIN.\n";
}

# Hardcoded Vina executable path (from your original script)
# Consider making this an argument as well for more flexibility,
# or ensure it's in the system PATH where the script runs.
my $vina_executable = "/mount/src/machine-learning/vina/vina_1.2.5_linux_x86_64";

print "--- Starting Vina Docking Process ---\n";
my $ligand_counter = 0;
for my $ligand_filepath_entry (@arr_file_paths) {
    chomp $ligand_filepath_entry; # Remove newline character from the path

    # Skip empty lines that might have been piped
    next if ($ligand_filepath_entry =~ /^\s*$/);
    $ligand_counter++;

    print "\n[$ligand_counter] Processing ligand: $ligand_filepath_entry\n";

    # Construct the Vina command
    # IMPORTANT: This command assumes that the receptor, output PDBQT (--out),
    # and log file (--log) are either defined *inside* your $config_file_path
    # or you are okay with Vina's default behavior (e.g., writing 'out.pdbqt'
    # and 'log.txt' in the current working directory, which will be overwritten
    # for each ligand if the config doesn't specify unique outputs).
    my $vina_command = "$vina_executable --config \"$config_file_path\" --ligand \"$ligand_filepath_entry\"";

    print "Executing: $vina_command\n";
    my $return_code = system($vina_command);

    if ($return_code == 0) {
        print "Vina completed successfully for $ligand_filepath_entry.\n";
    } else {
        my $exit_value  = $? >> 8;
        my $signal_num  = $? & 127;
        my $dumped_core = $? & 128;
        print "Vina FAILED for $ligand_filepath_entry. Exit code: $exit_value";
        print ", Signal: $signal_num" if $signal_num;
        print ", Core dumped" if $dumped_core;
        print "\n";
    }
    print "-------------------------------------\n";
}

if ($ligand_counter > 0) {
    print "\n--- Vina docking process finished for $ligand_counter ligand(s). ---\n";
} else {
    print "\n--- No ligands were processed. ---\n";
}
