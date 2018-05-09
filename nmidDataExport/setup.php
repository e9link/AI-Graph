<?php

/*
+-------------------------------------------------------------------------+
| Copyright:  2009-2013 by Urban-Software.de / Thomas Urban               |
|                                                                         |
| This program is free software; you can redistribute it and/or           |
| modify it under the terms of the GNU General Public License             |
| as published by the Free Software Foundation; either version 2          |
| of the License, or (at your option) any later version.                  |
|                                                                         |
| This program is distributed in the hope that it will be useful,         |
| but WITHOUT ANY WARRANTY; without even the implied warranty of          |
| MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           |
| GNU General Public License for more details.                            |
+-------------------------------------------------------------------------+
| - nmidDataExport - http://blog.network-outsourcing.de/                  |
+-------------------------------------------------------------------------+
*/


function plugin_nmidDataExport_install()
{
	api_plugin_register_hook( 'nmidDataExport', 'config_settings', 'nmidDataExport_config_settings', 'setup.php' );
	api_plugin_register_hook( 'nmidDataExport', 'console_after', 'nmidDataExport_console_after', 'setup.php' );
	api_plugin_register_hook( 'nmidDataExport', 'poller_bottom', 'nmidDataExport_check_dataExports', 'setup.php' );
	api_plugin_register_hook( 'nmidDataExport', 'graphs_action_array', 'plugin_nmidDataExport_graphs_action_array', 'setup.php' );
	api_plugin_register_hook( 'nmidDataExport', 'graphs_action_prepare', 'plugin_nmidDataExport_graphs_action_prepare', 'setup.php' );
	api_plugin_register_hook( 'nmidDataExport', 'graphs_action_execute', 'plugin_nmidDataExport_graphs_action_execute', 'setup.php' );
	nmidDataExport_setup_table_new();
}

function plugin_nmidDataExport_graphs_action_array( $action )
{
	$action[ 'plugin_nmidDataExport_graphs_export' ] = 'Automated Export - Add to Export';
	$action[ 'plugin_nmidDataExport_graphs_remove' ] = 'Automated Export - Remove graph(s)';
	return $action;
}

function plugin_nmidDataExport_graphs_action_prepare( $save )
{
	# globals used
	global $config, $colors;
	if ( preg_match( '/plugin_nmidDataExport_graphs_export/', $save[ "drp_action" ], $matches ) ) { /* nmidDataExport Server x */
		/* find out which (if any) hosts have been checked, so we can tell the user */
		if ( isset( $save[ "graph_array" ] ) ) {
			/* list affected hosts */
			print "<tr>";
			print "<td class='textArea' bgcolor='#" . $colors[ "form_alternate1" ] . "'>" .
				"<p>Are you sure you want to add the following graphs to the automated export?</p>" .
				"<p><ul>" . $save[ "graph_list" ] . "</ul></p>" .
				"</td>";
			print "</tr>";
		}
	}
	if ( preg_match( '/plugin_nmidDataExport_graphs_remove/', $save[ "drp_action" ] ) ) {
		if ( isset( $save[ "graph_array" ] ) ) {
			/* list affected hosts */
			print "<tr>";
			print "<td class='textArea' bgcolor='#" . $colors[ "form_alternate1" ] . "'>" .
				"<p>Are you sure you want to remove the following graphs from the automated export?</p>" .
				"<p><ul>" . $save[ "graph_list" ] . "</ul></p>" .
				"</td>";
			print "</tr>";
		}
	}
	return $save; # required for next hook in chain
}

function plugin_nmidDataExport_graphs_action_execute( $action )
{
	global $config;

	# it's our turn
	if ( preg_match( '/plugin_nmidDataExport_graphs_export/', $action, $matches ) ) { /* nmidDataExport Server x */
		if ( isset( $_POST[ "selected_items" ] ) ) {
			$selected_items = unserialize( stripslashes( $_POST[ "selected_items" ] ) );
			for ( $i = 0; ( $i < count( $selected_items ) ); $i++ ) {
				/* ================= input validation ================= */
				input_validate_input_number( $selected_items[ $i ] );
				/* ==================================================== */

				$data           = array();
				$data[ "lgid" ] = $selected_items[ $i ];

				db_execute( "INSERT into plugin_nmidDataExport_DataItems (lgid,rraid) VALUES(" . $data[ "lgid" ] . ",0);" );
			}
		}
	}
	if ( preg_match( '/plugin_nmidDataExport_graphs_remove/', $action ) ) { /* nmidDataExport Server x */
		if ( isset( $_POST[ "selected_items" ] ) ) {
			$selected_items = unserialize( stripslashes( $_POST[ "selected_items" ] ) );
			for ( $i = 0; ( $i < count( $selected_items ) ); $i++ ) {
				/* ================= input validation ================= */
				input_validate_input_number( $selected_items[ $i ] );
				/* ==================================================== */

				$data           = array();
				$data[ "lgid" ] = $selected_items[ $i ];

				db_execute( "DELETE FROM plugin_nmidDataExport_DataItems WHERE lgid=" . $data[ "lgid" ] );
			}
		}
	}
	return $action;
}

function plugin_nmidDataExport_uninstall()
{
	// Do any extra Uninstall stuff here
}

function plugin_nmidDataExport_check_config()
{
	// Here we will check to ensure everything is configured
	nmidDataExport_check_upgrade();
	return TRUE;
}

function plugin_nmidDataExport_upgrade()
{
	// Here we will upgrade to the newest version
	nmidDataExport_check_upgrade();
	return FALSE;
}

function plugin_nmidDataExport_version()
{
	return nmidDataExport_version();
}

function nmidDataExport_check_upgrade()
{
	// We will only run this on pages which really need that data ...
	$files = array( 'plugins.php' );
	if ( isset( $_SERVER[ 'PHP_SELF' ] ) && !in_array( basename( $_SERVER[ 'PHP_SELF' ] ), $files ) ) {
		return;
	}

	$current = nmidDataExport_version();
	$current = $current[ 'version' ];
	$old     = db_fetch_cell( "SELECT version FROM plugin_config WHERE directory='nmidDataExport'" );
	if ( $current != $old ) {
		nmidDataExport_setup_table( $old );
	}
}


function nmidDataExport_check_dependencies()
{
	global $plugins, $config;
	return TRUE;
}


function nmidDataExport_setup_table_new()
{
	global $config, $database_default;
	include_once( $config[ "library_path" ] . "/database.php" );

	// Check if the CereusReporting tables are present
	$s_sql = 'show tables from `' . $database_default . '`';
	$result = db_fetch_assoc( $s_sql ) or die ( mysql_error() );
	$a_tables = array();

	$sql = array();

	foreach ( $result as $index => $array ) {
		foreach ( $array as $table ) {
			$a_tables[ ] = $table;
		}
	}

	if ( !in_array( 'plugin_nmidDataExport_DataItems', $a_tables ) ) {
		// Create Report Table
		$data                 = array();
		$data[ 'columns' ][ ] = array( 'name' => 'lgid', 'type' => 'int', 'unsigned' => 'unsigned', 'NULL' => TRUE );
		$data[ 'columns' ][ ] = array( 'name' => 'rraid', 'type' => 'int', 'unsigned' => 'unsigned', 'NULL' => TRUE );
		$data[ 'keys' ][ ]    = array( 'name' => 'lgid', 'columns' => 'lgid' );
		$data[ 'primary' ]    = 'lgid';
		$data[ 'type' ]       = 'MyISAM';
		$data[ 'comment' ]    = 'nmidDataExport DataItems Export';
		api_plugin_db_table_create( 'nmidDataExport', 'plugin_nmidDataExport_DataItems', $data );
	}
}

function nmidDataExport_check_dataExports()
{
	global $config, $database_type, $database_default, $database_hostname, $database_username, $database_password, $database_port;
	include_once( $config[ "library_path" ] . "/database.php" );

	$fileCount = 0;

	$dir = dirname( __FILE__ );

	$a_ExportItems = db_fetch_assoc( "SELECT * FROM `plugin_nmidDataExport_DataItems`" );

	$fh_index = fopen( read_config_option( "nmid_de_exportDir" ).'/index.html', 'w' ) or die( "Can't open file" );
	fwrite($fh_index,'<h2>Exported Files</h2>'."\n");
	if ( read_config_option( "nmid_de_datatype" ) == 'JSON' ) {
		$data = array();
		$filename = 'devices.json';
		fwrite($fh_index,'<a href="'.$filename.'">'.$filename.'</a><br>'."\n");
		$fh = fopen( read_config_option( "nmid_de_exportDir" ).'/' . $filename, 'w' ) or die( "Can't open file" );

	}
	foreach ( $a_ExportItems as $s_ExportItem ) {
		$graph_info = db_fetch_row( "SELECT * FROM graph_templates_graph WHERE local_graph_id='" . $s_ExportItem[ "lgid" ] . "'" );


		/* for bandwidth, NThPercentile */
		$xport_meta = array();
		$graph_data_array = array();
		$graph_data_array["graph_start"] = time() - 86400;
		$graph_data_array["graph_end"] = time();
              

			/* Get graph export */
		$xport_array = @rrdtool_function_xport($s_ExportItem[ "lgid" ], $s_ExportItem[ "rraid" ], $graph_data_array, $xport_meta);
		$hostname = db_fetch_cell("SELECT hostname FROM host WHERE id = ".$xport_array["meta"]["host_id"]);
		$fileending = '.json';
		if ( read_config_option( "nmid_de_datatype" ) == 'CSV' ) {
			$fileending = '.csv';
		}

		/* Make graph title the suggested file name */
		if ( is_array( $xport_array[ "meta" ] ) ) {
		 	$filename = $hostname. '_' . $xport_array[ "meta" ][ "local_graph_id" ] . $fileending;
		}
		else {
			$filename = 'graph_export'.$fileending;
		}

		if (read_config_option("log_verbosity") >= POLLER_VERBOSITY_MEDIUM ) {
			cacti_log("Creating file '" .read_config_option( "nmid_de_exportDir" ).'/'. $filename . "'", FALSE, "nmidDataExport");
		}

		if ( read_config_option( "nmid_de_datatype" ) == 'CSV' ) {
			fwrite($fh_index,'<a href="'.$filename.'">'.$filename.'</a><br>'."\n");
			$fh = fopen( read_config_option( "nmid_de_exportDir" ).'/' . $filename, 'w' ) or die( "Can't open file" );
		}

		if ( is_array( $xport_array[ "meta" ] ) ) {
			$header = '"Date"';
			for ( $i = 1; $i <= $xport_array[ "meta" ][ "columns" ]; $i++ ) {
				$dataSource = $xport_array[ "meta" ][ "legend" ][ "col" . $i ];
				if ( ( $dataSource == 'In' ) || $dataSource == 'Out' ) {
					$header .= ',"' . $xport_array[ "meta" ][ "legend" ][ "col" . $i ] . '"';
				}
			}
			if ( read_config_option( "nmid_de_datatype" ) == 'CSV' ) {
                                fwrite( $fh, '"Title:","' . $xport_array[ "meta" ][ "title_cache"] . "\"\n" ); 
				fwrite( $fh, $header . "\n" );
			}
			else {
				// JSON
			}
		}
		if ( is_array( $xport_array[ "data" ] ) ) {
			foreach ( $xport_array[ "data" ] as $row ) {
				if ( read_config_option( "nmid_de_datatype" ) == 'CSV' ) {
					$data = '"' . date( "Y-m-d H:i:s", $row[ "timestamp" ] ) . '"';
				} else {
					$timestamp =date( "Y-m-d H:i:s", $row[ "timestamp" ] );
				}
				for ( $i = 1; $i <= $xport_array[ "meta" ][ "columns" ]; $i++ ) {
					$dataSource = $xport_array[ "meta" ][ "legend" ][ "col" . $i ];
					if ( ( $dataSource == 'In' ) || $dataSource == 'Out' ) {
						if ( read_config_option( "nmid_de_datatype" ) == 'CSV' ) {
							$data .= ',"' . number_format($row[ "col" . $i ], 2, '.', '')  . '"';
						}
						else {
							$data[ $hostname.'_'.$s_ExportItem[ "lgid" ] ][$timestamp][$dataSource] = number_format($row[ "col" . $i ], 2, '.', '');
						}
					}
				}
				if ( read_config_option( "nmid_de_datatype" ) == 'CSV' ) {
					fwrite( $fh, $data . "\n" );
				}
			}

		}
		if ( read_config_option( "nmid_de_datatype" ) == 'CSV' ) {
			fclose( $fh );
		}
		$fileCount++;
	}
	fclose( $fh_index );

	if ( read_config_option( "nmid_de_datatype" ) == 'JSON' ) {
		// JSON
		fwrite( $fh, json_encode($data) . "\n" );
		fclose( $fh );
	}
	if (read_config_option("log_verbosity") >= POLLER_VERBOSITY_MEDIUM ) {
		cacti_log("Exported " . $fileCount . " file(s)", FALSE, "nmidDataExport");
	}

}

function nmidDataExport_console_after()
{
	global $config, $plugins;
	nmidDataExport_setup_table();
}

function nmidDataExport_config_settings()
{
	global $tabs, $settings;
	$tabs[ "nmid" ] = "NMID";

	$temp = array(
		"nmid_de_header"   => array(
			"friendly_name" => "NMID - DataExport - General",
			"method"        => "spacer",
		),
		"nmid_de_datatype" => array(
			"friendly_name" => "Data Export Type",
			"description"   => "You can choose between CSV or JSON data format.",
			"method"        => "drop_array",
			"default"       => "CSV",
			"array"         => array(
				"CSV"  => "CSV",
				"JSON" => "JSON"
			)
		),
		"nmid_de_exportDir" =>  array(
			"friendly_name" => 'Export Directory',
			"description"   => 'The directory where to export the files.',
			"method"        => "textbox",
			"default"       => dirname(__FILE__) . '/export/',
			"max_length"    => 255
		)
	);

	if ( isset( $settings[ "nmid" ] ) ) {
		$settings[ "nmid" ] = array_merge( $settings[ "nmid" ], $temp );
	}
	else {
		$settings[ "nmid" ] = $temp;
	}
}

function nmidDataExport_version()
{
	return array( 'name'     => 'nmidDataExport',
	              'version'  => '1.01',
	              'longname' => 'NMID DataExport Plugin',
	              'author'   => 'Thomas Urban',
	              'homepage' => 'http://blog.network-outsourcing.de/support/nmid-plugins-support/',
	              'email'    => 'nmid@urban-software.de',
	              'url'      => 'http://urban-software.de/'
	);
}

function nmidDataExport_setup_table()
{
	$version_info = nmidDataExport_version();
	db_execute( 'UPDATE plugin_config SET version = "' . $version_info[ 'version' ] . '" WHERE directory = "nmidDataExport"' );
	nmidDataExport_setup_table_new();
}



?>
