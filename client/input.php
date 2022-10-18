<?php
include ("header.php");
include ("err_handle.php");
input_check();
?>

 <script type="text/javascript" src="client.js"></script>

<br>
    <b><img src=./img/notice.png border=0 width=20 height=20><font color=red> <?php echo $ini['notice'] ?></font></b><br>
<br>
<form action="output.php" method="post">
    <b><img src=./img/circle.png border=0 width=10 height=10></img> Model Name:</b> <?php echo $ini['model_name'] ?><br>
    <b><img src=./img/circle.png border=0 width=10 height=10></img> Supported Languages:</b> C/C++, Python, Java, JavaScript, Ruby, TypeScript, and Go.<br>
    <b><img src=./img/circle.png border=0 width=10 height=10></img> Temperature:</b>
    <select name="temperature">
        <option value="0.1" selected>0.1</option>
        <option value="0.2">0.2</option>
        <option value="0.3">0.3</option>
        <option value="0.4">0.4</option>
        <option value="0.5">0.5</option>
        <option value="0.2">0.6</option>
        <option value="0.6">0.2</option>
        <option value="0.7">0.7</option>
        <option value="0.8">0.8</option>
        <option value="0.9">0.9</option>
    </select>
    <b><img src=./img/circle.png border=0 width=10 height=10></img> Token Length:</b>
    <select name="token">
        <option value="50" selected>50</option>
        <option value="100">100</option>
        <option value="200">200</option>
        <option value="300">300</option>
        <option value="400">400</option>
        <option value="500">500</option>
        <option value="1000">1000</option>
        <option value="1500">1500</option>
        <option value="2000">2000</option>
        <option value="2025">2025</option>
    </select>
    <br>
    <b><img src=./img/circle.png border=0 width=10 height=10></img> Prompt Data:</b>
    <a href="javascript:widow_popup('help.php', 'popup');"><img src=./img/help.png width=30 height=30 border=0></img></a>
    <br>
    <textarea name="prompt" class='kb-source' rows="10" cols="100">
// Get the largest prime number that smaller than input
public static boolean isPrime(int n)
</textarea>
<br>

<p id="stapwatch"></p>
<p id="inference">Click me to generate source code automatically.</p>

<input type="submit" onclick="progressbar(); return true;" value="submit">
</form>
