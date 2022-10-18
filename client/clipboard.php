<link href="./style.css" rel="stylesheet" type="text/css">

<script type="text/javascript" src="client.js"></script>

<h3>Add the following javascript code to your page:</h3>

<p class="kb-source"><span id="sc">&lt;script&gt;<br>function copy_to_clipboard(id)<br>{<br>
var r = document.createRange();<br>r.selectNode(document.getElementById(id));<br>
window.getSelection().removeAllRanges();<br>window.getSelection().addRange(r);<br>
document.execCommand('copy');<br>window.getSelection().removeAllRanges();<br>
}<br>&lt;/script&gt;</span></p>

<p><a class="butt butt-blue" href="#" onclick="copy_to_clipboard('sc');return false;">Copy Code to Clipboard</a></p>
<p>&nbsp;</p>


