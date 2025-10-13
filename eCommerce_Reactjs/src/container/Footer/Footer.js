import React from 'react';

function Footer(props) {
  return (
    <div>
      <footer className="footer-area section_gap">
        <div className="container">
          <div className="row">
            <div className="col-lg-2 col-md-6 single-footer-widget">
              <h4>Top Products</h4>
              <ul>
                <li><a href="#">Managed Website</a></li>
                <li><a href="#">Manage Reputation</a></li>
                <li><a href="#">Power Tools</a></li>
                <li><a href="#">Marketing Service</a></li>
              </ul>
            </div>
            <div className="col-lg-2 col-md-6 single-footer-widget">
              <h4>Quick Links</h4>
              <ul>
                <li><a href="#">Jobs</a></li>
                <li><a href="#">Brand Assets</a></li>
                <li><a href="#">Investor Relations</a></li>
                <li><a href="#">Terms of Service</a></li>
              </ul>
            </div>
            <div className="col-lg-2 col-md-6 single-footer-widget">
              <h4>Features</h4>
              <ul>
                <li><a href="#">Jobs</a></li>
                <li><a href="#">Brand Assets</a></li>
                <li><a href="#">Investor Relations</a></li>
                <li><a href="#">Terms of Service</a></li>
              </ul>
            </div>
            <div className="col-lg-2 col-md-6 single-footer-widget">
              <h4>Resources</h4>
              <ul>
                <li><a href="#">Guides</a></li>
                <li><a href="#">Research</a></li>
                <li><a href="#">Experts</a></li>
                <li><a href="#">Agencies</a></li>
              </ul>
            </div>

          </div>
        </div>
      </footer>

    </div>
  );
}

export default Footer;