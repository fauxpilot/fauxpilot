<head>
<meta content="text/html; charset=utf-8" http-equiv="Content-Type">
<meta content="IE=edge" http-equiv="X-UA-Compatible">
<meta content="width=1200" name="viewport">
<link href="./style.css" rel="stylesheet" type="text/css">
</head>

<br>
<img src=img/fauxpilot.png border=0></img><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<?php
$ini = parse_ini_file ("./conf/config.ini");
echo "<font color=black>(". $ini['ip'] . ")</font>";
echo "<br>";
?>
