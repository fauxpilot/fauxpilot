/* java script */
let startTime;
let elapsedTime;

function time_to_string(time) {
    let diffInHrs = time / 3600000;
    let hh = Math.floor(diffInHrs);
  
    let diffInMin = (diffInHrs - hh) * 60;
    let mm = Math.floor(diffInMin);
  
    let diffInSec = (diffInMin - mm) * 60;
    let ss = Math.floor(diffInSec);
  
    let diffInMs = (diffInSec - ss) * 100;
    let ms = Math.floor(diffInMs);
  
    let formattedMM = mm.toString().padStart(2, "0");
    let formattedSS = ss.toString().padStart(2, "0");
    let formattedMS = ms.toString().padStart(2, "0");

    return `${formattedMM}:${formattedSS}`;
}

function my_timer() {
    elapsedTime = Date.now() - startTime;
    document.getElementById("stapwatch").innerHTML = "Elapsed time: " + time_to_string(elapsedTime);
}

function progressbar() {
    // stop watch
    setInterval(my_timer, 1000);
    startTime = Date.now();

    // progress bar (with image)
    document.getElementById("inference").innerHTML = "<img src=./img/progress.gif width=250 height=150 border=0></img><br><font color=red>It is being handled. Please wait for a moment.!!!</font>";
    window.scrollBy(0, 100);
}

function widow_popup(url, name){
    var options = 'top=10, left=10, width=500, height=600, status=no, menubar=no, toolbar=no, resizable=no';
    window.open(url, name, options);
}


function copy_to_clipboard(id) {
    var r = document.createRange();
    r.selectNode(document.getElementById(id));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(r);
    document.execCommand('copy');
    window.getSelection().removeAllRanges();
}

