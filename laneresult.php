
<!DOCTYPE HTML>


<html>
	<head>
	<meta charset="utf-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge">
	<title>Traffic Sign Analyzer</title>
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<meta name="description" content="Traffic Sign Analyzer" />
	<meta name="keywords" content="Traffic,Sign,lane,Analyzer" />
	<meta name="author" content="Anoop" />

  	<!-- Facebook and Twitter integration -->
	<meta property="og:title" content=""/>
	<meta property="og:image" content=""/>
	<meta property="og:url" content=""/>
	<meta property="og:site_name" content=""/>
	<meta property="og:description" content=""/>
	<meta name="twitter:title" content="" />
	<meta name="twitter:image" content="" />
	<meta name="twitter:url" content="" />
	<meta name="twitter:card" content="" />

	<link href="https://fonts.googleapis.com/css?family=Raleway:100,300,400,700" rel="stylesheet">
	
	<!-- Animate.css -->
	<link rel="stylesheet" href="css/animate.css">
	<!-- Icomoon Icon Fonts-->
	<link rel="stylesheet" href="css/icomoon.css">
	<!-- Themify Icons-->
	<link rel="stylesheet" href="css/themify-icons.css">
	<!-- Bootstrap  -->
	<link rel="stylesheet" href="css/bootstrap.css">

	<!-- Magnific Popup -->
	<link rel="stylesheet" href="css/magnific-popup.css">

	<!-- Owl Carousel  -->
	<link rel="stylesheet" href="css/owl.carousel.min.css">
	<link rel="stylesheet" href="css/owl.theme.default.min.css">

	<!-- Theme style  -->
	<link rel="stylesheet" href="css/style.css">

	<!-- Modernizr JS -->
	<script src="js/modernizr-2.6.2.min.js"></script>
	<!-- FOR IE9 below -->
	<!--[if lt IE 9]>
	<script src="js/respond.min.js"></script>
	<![endif]-->
	
	</head>
	<body>
		
	<div class="gtco-loader"></div>
	
	<div id="page">

		<nav class="gtco-nav" role="navigation">
			<div class="gtco-container">
				
				<div class="row">
					<div class="col-sm-2 col-xs-12">
						<div id="gtco-logo"><a href="index.html"><img src="images/sign.png" width="40px" height="40px"></a></div>
					</div>
					<div class="col-xs-10 text-right menu-1">
						<ul>
							<li class="active"><a href="index.html">Home</a></li>
							<li><a href="about.html">About</a></li>
							<li class="has-dropdown">
								<a href="services.html">Services</a>
								<ul class="dropdown">
									<li><a href="tt.html">Training & Testing</a></li>
									<li><a href="#">Lane detection</a></li>
								</ul>
							</li>
							<li class="has-dropdown">
								<a href="#">Results</a>
								<ul class="dropdown">
									<li><a href="#">Traffic Sign</a></li>
									<li><a href="#">Lane</a></li>
								</ul>
							</li>
							<!--<li><a href="portfolio.html">Portfolio</a></li>
							<li><a href="contact.html">Contact</a></li>-->
						</ul>
					</div>
				</div>
				
			</div>
		</nav>

		<footer id="gtco-footer" class="gtco-section" role="contentinfo">
		<div class="gtco-container">
				<div class="row row-pb-md">
					<div class="col-md-8 col-md-offset-2 gtco-cta text-center">
						<h3>TRAINING RESULT<br></h3>
						
					</div>
				</div>
				
					
					
					<div class="col-md-6 gtco-footer-subscribe">
							
							<?php

								$dirname = "output_img/";
								$images = glob($dirname."*.png");

								foreach($images as $image) {
    								echo '<img src="'.$image.'" width=1000/><br /><br />';
								}	

							?>	
							<center>
								<video loop autoplay width="80%" height="60%"  >
  									<source src="test_videos_output/solidWhiteRight.mp4" type="video/mp4">
  								</video><br><br /><br />
  								<video loop autoplay width="80%" height="60%"  >
  									<source src="test_videos_output/solidYellowLeft.mp4" type="video/mp4">
  								</video>		
  								</center>		
						
					</div>
				
			</div>
			<div class="gtco-copyright" >
				<div class="gtco-container">
					
					<div class="row">
						<div class="col-md-6 text-left">
							<p><small>&copy; 2018 All Rights Reserved. </small></p>
						</div>
						<div class="col-md-6 text-right">
							<p><small>SDMIT </small> </p>
						</div>
					</div>
				</div>
			</div>
		</footer>

	</div>

	<div class="gototop js-top">
		<a href="#" class="js-gotop"><i class="icon-arrow-up"></i></a>
	</div>
	
	<!-- jQuery -->
	<script src="js/jquery.min.js"></script>
	<!-- jQuery Easing -->
	<script src="js/jquery.easing.1.3.js"></script>
	<!-- Bootstrap -->
	<script src="js/bootstrap.min.js"></script>
	<!-- Waypoints -->
	<script src="js/jquery.waypoints.min.js"></script>
	<!-- countTo -->
	<script src="js/jquery.countTo.js"></script>
	<!-- Carousel -->
	<script src="js/owl.carousel.min.js"></script>
	<!-- Magnific Popup -->
	<script src="js/jquery.magnific-popup.min.js"></script>
	<script src="js/magnific-popup-options.js"></script>

	
	
	<!-- Main -->
	<script src="js/main.js"></script>

	</body>
</html>


