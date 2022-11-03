<?php
// Notify the browser about the type of the file using the header function
header('Content-type: text/html');

include ("header.php");
include ("err_handle.php");
?>

<br>
<form>
<input type="button" value="Go back!" onclick="window.location.assign('./input.php')">
</form>
</body>
</html>
<?php
// Specify input data
$token = (int) $_POST["token"];
$prompt = $_POST["prompt"];
$temperature = (float) $_POST["temperature"];

// cURL PHP API test
function cURLcheckBasicFunctions() {
    if( !function_exists("curl_init") &&
        !function_exists("curl_setopt") &&
        !function_exists("curl_exec") &&
        !function_exists("curl_close") ) return false;
    else return true;
}

if( !cURLcheckBasicFunctions() ) {
    return "UNAVAILABLE: cURL Basic Functions";
}

// Collect objects
$data_array = [
	'prompt' => $prompt,
	'max_tokens' => $token,
	'temperature' => $temperature,
	'stop' => ["\n\n"]
];

// Initialize a new cURL session
$url = "http://" . $ini['ip'] . ":".$ini['port'] . "/v1/engines/codegen/completions";
$curl = curl_init($url);

// Set the CURLOPT_RETURNTRANSFER option to true
curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);

// Set the CURLOPT_POST option to true for POST request
curl_setopt($curl, CURLOPT_POST, true);

// Set the request data as JSON using json_encode function
curl_setopt($curl, CURLOPT_POSTFIELDS,  json_encode($data_array));

// Set custom headers for RapidAPI Auth and Content-Type header
curl_setopt($curl, CURLOPT_HTTPHEADER, [
  'Accept: application/json',
  'Content-Type: application/json'
]);


curl_setopt($curl, CURLOPT_SSL_VERIFYPEER, false);

// Execute cURL request with all previous settings
$response = curl_exec($curl);

if ( $ini['debug'] == 1 ) {
    echo "<table><tr><td>";
    echo "<div class='c-warning'>";
    echo '<b><font color=red>* Dump messages for debugging:</font></b><br>';
    var_dump($response);
    echo '<br>';
    echo '</div>';
    echo "</td></tr></table>";
}


// Define recursive function to extract nested values
function print_values($arr) {
    global $count;
    global $values;
    
    // Check input is an array
    if(!is_array($arr)){
        die("ERROR: Input is not an array");
    }
    
    // Loop through array, if value is itself an array recursively call the
    // function else add the value found to the output items array,
    // and increment counter by 1 for each value found
    foreach($arr as $key=>$value){
        if(is_array($value)){
            print_values($value);
        } else{
            $values[] = $value;
            $count++;
        }
    }
    
    // Return total count and values found in array
    return array('total' => $count, 'values' => $values);
}


// Decode JSON data into PHP associative array format
$arr = json_decode($response, true);
 
$output = $arr["choices"][0]["text"];
$output = str_replace(array('\r\n', '\n\r', '\n', '\r', ';'), ';', $output);
echo "<table><tr><td>";
echo "<pre>"
?>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<?php
echo "<b>Temperature:</b>" . $temperature . "<br>";
echo "<b>Token:</b>" . $token . "<br>";
echo "<b>URL:</b>" . $url . "<br>";
echo "<b>Request ID:</b>" . $arr["id"] . "<br>";
echo "<b>Code Auto Completion:</b>" . "<br>";
echo "</pre>";
echo "<pre>";
echo "<div class='kb-source'>";
echo "<span id='sc'>";
echo "<font color=blue>" . $prompt . "</font>" ; 
echo "<font color=red>" . $output . "</font>" ; 
echo "</span>";
echo "</div>";
echo "</pre>";
echo "<br>";
echo "</td></tr></table>";
output_check($output);

// Close cURL session
curl_close($curl);
?>
<a class="butt butt-blue" href="#" onclick="copy_to_clipboard('sc');return false;">Copy Code to Clipboard</a>
<p>&nbsp;</p>

