function colorbar_relocate() {

    var winTop = $(window).scrollTop();
    var passTop = $('#content-upper-graphs').offset().top;
    var divTop = $('#content-colorbar').offset().top;
    
    if (winTop > (passTop - 25)) {
        $('#content-colorbar').addClass('stick');
    } else {
        $('#content-colorbar').removeClass('stick');
    }
}

$(document).ready(function() {
    $(window).scroll(colorbar_relocate);
    colorbar_relocate();
});

