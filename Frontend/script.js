function init() {
  gsap.registerPlugin(ScrollTrigger);

  const locoScroll = new LocomotiveScroll({
    el: document.querySelector(".main"),
    smooth: true,
  });

  locoScroll.on("scroll", ScrollTrigger.update);

  ScrollTrigger.scrollerProxy(".main", {
    scrollTop(value) {
      return arguments.length
        ? locoScroll.scrollTo(value, 0, 0)
        : locoScroll.scroll.instance.scroll.y;
    },
    getBoundingClientRect() {
      return {
        top: 0,
        left: 0,
        width: window.innerWidth,
        height: window.innerHeight,
      };
    },

    pinType: document.querySelector(".main").style.transform
      ? "transform"
      : "fixed",
  });
}
init()

var crsr=document.querySelector(".cursor")
var main=document.querySelector(".main")
document.addEventListener("mousemove",function(dets){
    crsr.style.left=dets.x+20+"px"
    crsr.style.top=dets.y+20+"px"
})
var tl = gsap.timeline({
  scrollTrigger: {
    trigger: ".page1 h1",
    scroller: ".main",
    start: "top 40%",
    end: "top 10",
    scrub: 3,
  },
});
tl.to(
  ".page1 h1",
  {
    x: -100,
  },
  "anim"
);

tl.to(
  ".page1 h2",
  {
    x: 100,
  },
  "anim"
);

tl.to(
  ".page1 video",
  {
    width: "90%",
  },
  "anim"
);

var tl2 = gsap.timeline({
  scrollTrigger: {
    trigger: ".page1 h1",
    scroller: ".main",
    start: "top -97%",
    end: "top -130%",
    scrub: 3,
  },
});
tl2.to(".main", {
  backgroundColor: "#fff",
});


var tl3 = gsap.timeline({
  scrollTrigger: {
    trigger: ".page1 h1",
    scroller: ".main",
    start: "top -350%",
    end: "top -300%",
    scrub: 3,
  },
});

tl3.to(".main", {
  backgroundColor: "#0F0D0D",
});

var tl4 = gsap.timeline({
  scrollTrigger: {
    trigger: ".page1 h1",
    scroller: ".main",
    start: "top -670%",
    end: "top -100%",
    scrub: 3,
  },
});

tl4.to(".footer", {
  backgroundColor: "#EDBFFF",
});


// var boxes=document.querySelectorAll(".box")
// boxes.forEach(function(elem){
//       elem.addEventListener("mouseenter",function(){
//         var att=elem.getAttribute("data-image")
//         crsr.style.width="300px"
//         crsr.style.height="250px"
//         crsr.style.borderRadius="0"
//         crsr.style.backgroundImage=`url(${att})`
//       })
//       elem.addEventListener("mouseleave",function(){
//         elem.style.backgroundColor = "transparent"
//         crsr.style.width="20px"
//         crsr.style.height="20px"
//         crsr.style.borderRadius="50%"
//         crsr.style.backgroundImage=`none`
//       })
// })
var boxes = document.querySelectorAll(".box");

boxes.forEach(function(elem) {
    elem.addEventListener("mouseenter", function() {
        var att = elem.getAttribute("data-image");
        var imageUrl;

        // Check if the attribute contains a local file path or a URL
        if (att.startsWith("http") || att.startsWith("www")) {
            imageUrl = `url(${att})`; // If it's a URL
        } else {
            imageUrl = `url(${att})`; // If it's a local file path
        }

        crsr.style.width = "350px";
        crsr.style.height = "250px";
        crsr.style.borderRadius = "0";
        crsr.style.backgroundImage = imageUrl; // Set the background image
    });

    elem.addEventListener("mouseleave", function() {
        elem.style.backgroundColor = "transparent";
        crsr.style.width = "20px";
        crsr.style.height = "20px";
        crsr.style.borderRadius = "50%";
        crsr.style.backgroundImage = "none"; // Remove the background image
    });
});


var Y=document.querySelectorAll("#Y")
var purple=document.querySelector("#purple")
Y.forEach(function(elem){
  elem.addEventListener("mouseenter",function(){
    purple.style.visibility="visible"
    purple.style.opacity="1"
  })
  elem.addEventListener("mouseleave",function(){
    purple.style.visibility="hidden"
    purple.style.opacity="0"
  })
})

var GN=document.querySelectorAll("#GN")
var maroon=document.querySelector("#maroon")
GN.forEach(function(elem){
  elem.addEventListener("mouseenter",function(){
    maroon.style.visibility="visible"
    maroon.style.opacity="1"
  })
  elem.addEventListener("mouseleave",function(){
    maroon.style.visibility="hidden"
    maroon.style.opacity="0"
  })
})

var Mem=document.querySelectorAll("#Mem")
var Salmon=document.querySelector("#salmon")
Mem.forEach(function(elem){
  elem.addEventListener("mouseenter",function(){
    Salmon.style.visibility="visible"
    Salmon.style.opacity="1"
  })
  elem.addEventListener("mouseleave",function(){
    Salmon.style.visibility="hidden"
    Salmon.style.opacity="0"
  })
})

function redirectToLocalhost() {
  window.location.href = 'http://127.0.0.1:5000/';
}

function page2Animation(){
  gsap.from(".page2",{
    y:120,
    stagger:0.2,
    duration:1,
    scrollTrigger:{
      trigger:".page2",
      scroller:".main",
      start:"top 47%",
      end:"top 46%",
      markers:false,
      scrub:2

    }
  })
}
page2Animation()
