$(document).ready(function() {
    $(window).scroll(function() {
        if ($(this).scrollTop() > 50) {
            $('.navbar').addClass('scrolled');
        } else {
            $('.navbar').removeClass('scrolled');
        }
    });

    // Function to handle machine selection
    function selectMachine(machine) {
        // Replace the machine name with underscores for consistency with Python dictionary keys
        machine = machine.replace(/ /g, '_');

        // Fetch components for the selected machine via AJAX
        fetch(`/get_components/${machine}`)
            .then(response => response.json())
            .then(components => {
                console.log("Received Compo data");
                displayMachineDetails(machine, components);
            })
            .catch(error => console.error('Error fetching components:', error));
    }

    // Function to display machine details in the overlay
    function displayMachineDetails(machine, components) {
        const overlay = document.getElementById('overlay');
        const machineDetails = document.getElementById('machine-details');
        const machineName = document.getElementById('machine-name');
        const componentsDiv = document.getElementById('components');
        const graphsDiv = document.getElementById('graphs');

        // Set the machine name in the overlay header
        machineName.textContent = machine.replace(/_/g, ' ');

        // Clear previous component and graph displays
        componentsDiv.innerHTML = '';
        graphsDiv.innerHTML = '';
        console.log("Compo data");
        // Iterate over components to display in the form
        components.forEach(component => {
            const componentDiv = document.createElement('div');
            componentDiv.classList.add('component');
            componentDiv.onclick = () => fetchGraphs(machine, component);
            console.log("Compo datas");
            const componentImg = document.createElement('img');
            componentImg.src = `/static/${component}.png`; // Adjust the image path as needed
            componentImg.alt = `${component} Image`;
            componentDiv.appendChild(componentImg);

            const componentName = document.createElement('p');
            componentName.textContent = component;
            componentDiv.appendChild(componentName);

            componentsDiv.appendChild(componentDiv);
        });

        // Display the overlay
        overlay.style.display = 'block';
    }

    // Function to fetch graphs for the selected component
    function fetchGraphs(machine, component) {
        fetch(`/fetch_data?machine=${machine}&component=${component}`)
            .then(response => response.json())
            .then(data => {
                console.log('Received graph data:', data);
                displayGraphs(data);
            })
            .catch(error => console.error('Error fetching graphs:', error));
    }

    // Function to display graphs and threshold crossings
    function displayGraphs(data) {
        const graphsDiv = document.getElementById('graphs');
        graphsDiv.innerHTML = '';

        data.forEach(item => {
            const parameterDiv = document.createElement('div');
            parameterDiv.classList.add('parameter');

            // Display parameter name
            const parameterName = document.createElement('p');
            parameterName.textContent = item.parameter;
            parameterDiv.appendChild(parameterName);

            // Display graph image for the parameter
            const graphImage = document.createElement('img');
            graphImage.src = 'data:image/png;base64,' + item.plot_url;
            graphImage.alt = `${item.parameter} Graph`;
            parameterDiv.appendChild(graphImage);

            // Display threshold crossings
            const thresholdList = document.createElement('ul');
            item.threshold_crossings.forEach(crossing => {
                const thresholdItem = document.createElement('li');
                thresholdItem.textContent = crossing;
                thresholdList.appendChild(thresholdItem);
            });
            parameterDiv.appendChild(thresholdList);

            graphsDiv.appendChild(parameterDiv);
        });
    }

    // Function to close the overlay
    function closeOverlay() {
        const overlay = document.getElementById('overlay');
        overlay.style.display = 'none';
    }

    // Attach the function to the window object for global access
    window.selectMachine = selectMachine;
    window.closeOverlay = closeOverlay;
});
