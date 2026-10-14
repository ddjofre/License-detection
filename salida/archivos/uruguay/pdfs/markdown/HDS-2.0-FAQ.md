<!-- Start of picture text -->
salesforce<br><!-- End of picture text -->

# **Hébergeur de Données de Santé (HDS) Version 2.0** 

## **Frequently Asked Questions (FAQs)** 

At Salesforce, Trust is our #1 value and the protection of our customers' data is paramount. We know that many organizations in the healthcare and life sciences industries have questions about  the French Public Health Code and the Hébergeur de Données de Santé (HDS) certification. To help you develop a better understanding of HDS version 2.0 in the context of Salesforce's online services and support your HDS compliance journey, we have created this FAQ. 

### **Overview** 

This document provides answers to frequently asked questions about Salesforce's online services that are covered by our HDS certifications (which are called in this document " **HDS Covered Services** "). This document does not provide  legal advice and the information we present may not take into account future changes in laws and regulations. We urge you to consult with your own legal counsel to familiarize yourself with the requirements that govern your specific situation. 

### **What is HDS? Who is subject to HDS?** 

The hosting of personal health data in France is regulated by Article L. 1111-8 of the Public Health Code (Code de la santé publique). It requires that any natural or legal person who hosts personal health data collected during prevention, diagnosis, care, or social and medico-social tracking activities must use an HDS-certified provider. 

HDS 2.0 is the latest iteration of this certification framework published by the Agence du Numérique en Santé (ANS). It updates the previous standards to align with the latest ISO/IEC 27001:2022 requirements, introduces stricter guidelines on data residency (such as mandatory storage within the European Economic Area), and reinforces the

management of subcontractors and data transfers. 

# **Does HDS apply to Salesforce?** 

Yes, we are subject to HDS as a certified cloud service provider hosting French personal health data (referred to “ **French Health Data** ” in this  document) on behalf of our customers that: (1) are subject to French health data regulations; and (2) choose to submit French Health Data to our HDS Covered Services. 

# **What HDS Covered Services does Salesforce provide?** 

Customers can verify the specific Salesforce services that are in scope for our HDS certification by downloading our official HDS certificate, available on our <u>Salesforce Compliance Portal .</u> 

# **Does Salesforce comply with HDS?** 

Yes, **with respect to the HDS Covered Services** . In  provisioning and operating these services, we comply with the required provisions of the HDS 2.0 certification and the applicable European and French data protection laws regarding the hosting of French Health Data. As noted above, Customers can verify which Salesforce services are in scope for our HDS certification by downloading our official HDS certificate on the <u>Salesforce Compliance Portal .</u> 

# **How does Salesforce comply with the data residency requirements under HDS 2.0?** 

HDS 2.0 places a strong emphasis on data residency and transparency. French Health Data will be hosted exclusively in the EEA at sites covered by relevant HDS certificates. Access to and processing of French Health Data by Salesforce from outside the EEA will be subject to appropriate transfer mechanisms as described in our Data Processing Addendum (DPA). 

Furthermore, to meet specific HDS requirements (Requirements No. 29 and 30) regarding the risk of data access from outside the EEA, Salesforce provides an _HDS Representation of Guarantees_ available on the  Salesforce Compliance  Portal . 

# **Do we enter into specific contracts for HDS with our customers?** 

Yes. In addition to our existing contractual framework (in particular the Main Services Agreement (“MSA”) and Data Processing Agreement (“DPA”)), Salesforce has developed a **HDS Hosting Services Addendum (“HHSA”)** . 

The HHSA is available on request and is incorporated into the MSA and sets out the HDS-specific terms governing Salesforce’s hosting of French Health Data under our HDS Covered Services.

The HHSA applies exclusively to **those HDS Covered Services** hosted within the European Economic Area. Furthermore, these commitments apply to production environments and explicitly exclude non-production environments (such as sandboxes), where French Health Data should not be hosted. 

If you wish to request the HHSA please engage with your Account Executive. 

# **How  does  using  Salesforce  Services  comply  with  HDS  security requirements?** 

Salesforce complies with the provisions of the HDS framework by protecting Customer Data, including French Health Data submitted to our services, through  robust administrative, technical, and physical safeguards. However, security and compliance under HDS 2.0 is a shared responsibility and Customers have a role to play in ensuring that their environment is set up appropriately to handle the data being uploaded to Salesforce Services. 

## **A. The Shared Responsibility Framework** 

We offer our customers a broad array of security features that they can implement in connection with their use of the HDS Covered Services. Our product and service features provide our customers with flexibility when complying with stringent security requirements such as those mandated by HDS. 

Importantly, our customers should be aware that while we comply with HDS in provisioning and operating the HDS Covered Services as a certified host, it is the sole responsibility of our customers to ensure their compliance with applicable laws concerning their particular use of the HDS Covered Services. In particular, when using the HDS Covered Services, customers should be aware of their role and the actions that they must take to protect their information: 

- It is the sole responsibility of the customer to ensure that their use of the HDS Covered Services complies with applicable laws. 

- Customers must appropriately configure their security settings and apply the security controls they deem appropriate for the data being handled through Salesforce Services. 

- Customers must strictly manage access rights and conduct their own risk assessments in accordance with French law. 

- Customers  should  ensure  that  their  instance  is  provisioned  on  EEA  environments.  In the  event  that  a  customer  is  provisioned  in  other  jurisdictions,  customers  are responsible  for  contacting  their  Account  Executive  so  that  they  can  be  re-provisioned

# appropriately. 

# **B. Salesforce Internal Controls** 

To secure the underlying environment hosting French Health Data, Salesforce maintains strict internal controls: 

- Salesforce personnel generally do not access Customer Data, including French Health Data. In the limited scenarios where access is required for technical or support purposes, it is strictly controlled and fully logged. 

- Salesforce has in place Disaster Recovery and Business Continuity Plans described in the Documentation. Additional documentation and summaries are available on the <u>Salesforce Compliance Portal .</u> 

# **C. Customer-configured Security Controls** 

To enable customers to meet their strict compliance and risk management obligations such as those mandated by HDS, Salesforce makes available to its customers a comprehensive suite of security features as outlined in our Trust Documentation (in particular our Security, Privacy and Architecture Documentation). When accessing French Health Data, Customers can leverage these features to build their secure environments: 

- **Role-Based Access Control:** Implementing least-privilege  administration across Customers user base. 

- **Strong Authentication:** Enforcing multi-factor and  strong authentication mechanisms for administrative access. 

- **Traceability:** Utilizing comprehensive logging and  audit trails made available by Salesforce to track data modifications and user activity.