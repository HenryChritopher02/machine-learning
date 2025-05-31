#!/usr/bin/perl
use strict;
use warnings;
use File::Basename; # For basename() and dirname()
use File::Spec;     # For path manipulation, e.g., catfile()
use Cwd 'abs_path'; # To get absolute path if needed

# --- Script Arguments ---
# ARGV[0]: Absolute path to Vina executable
# ARGV[1]: Absolute path to the receptor PDBQT file (e.g., already copied to WORKSPACE_PARENT_DIR/proteinA.pdbqt by Python)
# ARGV[2]: Absolute path to the specific Vina config TXT file to be used
# ARGV[3]: Protein base name (e.g., "proteinA"), used for creating an output subdirectory

# --- Read and Validate Command-Line Arguments ---
my $vina_executable_path = $ARGV[0];
my $receptor_file_path   = $ARGV[1];
my $config_file_path     = $ARGV[2];
my $protein_base_name    = $ARGV[3];

if (not (defined $vina_executable_path && defined $receptor_file_path && defined $config_file_path && defined $protein_base_name)) {
    die "Usage: $0 <vina_executable_path> <receptor_pdbqt_path> <config_file_path> <protein_base_name>\n" .
        "       A path to a file containing a list of ligand PDBQT paths should be piped via STDIN.\n";
}

die "Error: Vina executable '$vina_executable_path' not found or not executable.\n" unless -f $vina_executable_path && -x $vina_executable_path;
die "Error: Receptor file '$receptor_file_path' not found or is not a regular file.\n" unless -f $receptor_file_path;
die "Error: Config file '$config_file_path' not found or is not a regular file.\n" unless -f $config_file_path;

print "--- Vina Screening Perl Script Initialized ---\n";
print "Vina Executable: $vina_executable_path\n";
print "Receptor File:   $receptor_file_path\n";
print "Config File:     $config_file_path\n";
print "Protein Base:    $protein_base_name\n";

# --- Read Path to Ligand List File from STDIN ---
print "Reading path to ligand list file from STDIN...\n";
my $ligand_list_file_path = <STDIN>;
die "Error: No path to ligand list file received via STDIN.\n" unless defined $ligand_list_file_path;
chomp $ligand_list_file_path;

# Resolve to absolute path if it's relative and exists (safer)
if (-f $ligand_list_file_path && !File::Spec->file_name_is_absolute($ligand_list_file_path)) {
    $ligand_list_file_path = abs_path($ligand_list_file_path);
}
die "Error: Ligand list file '$ligand_list_file_path' not found or is not a regular file.\n" unless -f $ligand_list_file_path;

print "Using Ligand List File: $ligand_list_file_path\n";

# --- Read Ligand File Paths from the List File ---
open(my $LIG_LIST_FH, '<', $ligand_list_file_path) or die "Error: Cannot open ligand list file '$ligand_list_file_path': $!\n";
my @arr_individual_ligand_paths = <$LIG_LIST_FH>;
close $LIG_LIST_FH;

if (not @arr_individual_ligand_paths) {
    die "Error: Ligand list file '$ligand_list_file_path' is empty.\n";
}

# --- Output Directory Setup ---
# Create an output subdirectory named after the protein_base_name
# This script runs with its CWD set to WORKSPACE_PARENT_DIR by the Python app.
# So, this directory will be created inside WORKSPACE_PARENT_DIR.
my $output_subdir_name = $protein_base_name; # e.g., "proteinA"
unless (-d $output_subdir_name) {
    mkdir $output_subdir_name or die "Error: Cannot create output subdirectory '$output_subdir_name': $!\n";
}
print "Outputs will be saved in subdirectory: ./$output_subdir_name (relative to script's CWD)\n";

# --- Start Docking Process ---
print "--- Starting Vina Docking Process for Protein: $protein_base_name ---\n";
my $ligand_counter = 0;
my $successful_dockings = 0;

for my $single_ligand_path (@arr_individual_ligand_paths) {
    chomp $single_ligand_path;
    next if ($single_ligand_path =~ /^\s*$/); # Skip empty lines
    $ligand_counter++;

    # Ensure ligand path is absolute for Vina, especially if paths in list file might be relative
    my $abs_single_ligand_path = $single_ligand_path;
    if (!File::Spec->file_name_is_absolute($single_ligand_path) && -f $single_ligand_path) {
        $abs_single_ligand_path = abs_path($single_ligand_path);
    } elsif (! -f $single_ligand_path) {
        print "Warning: Ligand file '$single_ligand_path' listed in '$ligand_list_file_path' not found. Skipping.\n";
        next;
    }

    print "\n[$ligand_counter] Processing Ligand: $abs_single_ligand_path\n";

    my $ligand_file_basename = basename($abs_single_ligand_path); # e.g., ligand1.pdbqt
    my ($ligand_name_only, $ligand_ext) = $ligand_file_basename =~ /^(.*?)(\.[^.]*)?$/; # Capture base and extension
    $ligand_name_only = $ligand_file_basename unless defined $ligand_ext; # Handle no extension

    # Define unique output file names within the protein-specific subdirectory
    my $output_docked_pdbqt = File::Spec->catfile($output_subdir_name, "${ligand_name_only}_${protein_base_name}_out.pdbqt");
    my $output_log_txt    = File::Spec->catfile($output_subdir_name, "${ligand_name_only}_${protein_base_name}_log.txt");

    # Construct the Vina command using the list form of system for safety with paths
    my @vina_command_args = (
        $vina_executable_path,
        '--receptor', $receptor_file_path,
        '--config',   $config_file_path,
        '--ligand',   $abs_single_ligand_path,
        '--out',      $output_docked_pdbqt
        # Add other Vina parameters if necessary (e.g., --cpu, --exhaustiveness, if not in config)
    );

    print "Executing: " . join(" ", @vina_command_args) . "\n";
    my $return_code = system(@vina_command_args); # Use list form of system

    if ($return_code == 0) {
        print "Vina completed successfully for $ligand_file_basename.\n";
        $successful_dockings++;
    } else {
        my $exit_value  = $? >> 8;
        my $signal_num  = $? & 127;
        my $dumped_core = $? & 128;
        print "Vina FAILED for $ligand_file_basename. Exit code: $exit_value";
        print ", Signal: $signal_num" if $signal_num;
        print ", Core dumped" if $dumped_core;
        print "\n";
    }
    print "-------------------------------------\n";
}

if ($ligand_counter > 0) {
    print "\n--- Vina docking process finished for protein '$protein_base_name'. ---\n";
    print "Attempted to process $ligand_counter ligand(s).\n";
    print "Successfully docked $successful_dockings ligand(s).\n";
} else {
    print "\n--- No ligands were found in the list file '$ligand_list_file_path' to process for protein '$protein_base_name'. ---\n";
}
