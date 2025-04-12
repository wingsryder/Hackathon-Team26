// JavaScript enhancements for Streamlit

// Add smooth scrolling for internal links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Highlight table rows on hover
document.querySelectorAll('table tr').forEach(row => {
    row.addEventListener('mouseover', () => {
        row.style.backgroundColor = '#f2f2f2';
    });
    row.addEventListener('mouseout', () => {
        row.style.backgroundColor = '';
    });
});

// Add a fade-in effect to headers
document.querySelectorAll('h1, h2, h3').forEach(header => {
    header.style.opacity = 0;
    header.style.transition = 'opacity 1s';
    setTimeout(() => {
        header.style.opacity = 1;
    }, 500);
});
