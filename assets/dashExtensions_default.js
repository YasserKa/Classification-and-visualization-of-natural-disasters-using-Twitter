window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
            {
                return ['Aarhus'].includes(feature.properties.name);
            }
        }
    }
});