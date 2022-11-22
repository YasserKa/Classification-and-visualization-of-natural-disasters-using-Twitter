from unittest import TestCase

from flood_detection.predict.extract_location import GeoCoder, GeoCodersEnum, Transform


class TestExtractLocation:
    def test_token_extractor(self) -> None:

        model = Transform("KBLab/bert-base-swedish-cased-ner")
        geocoder = GeoCoder(GeoCodersEnum.NOMINATIM.value)

        expected_tokens = [
            {
                "entity": "ORG",
                "end": 27,
                "index": 4,
                "score": 0.99900633,
                "start": 13,
                "word": "länsstyrelsens",
            },
            {
                "end": 86,
                "entity": "LOC",
                "index": 16,
                "score": 0.99568707,
                "start": 77,
                "word": "Uddevalla",
            },
            {
                "end": 98,
                "entity": "LOC",
                "index": 18,
                "score": 0.9959202,
                "start": 88,
                "word": "Ljungskile",
            },
        ]

        expected_locations_tokens = {
            "Uddevalla": {"ner_score": 0.99568707},
            "Ljungskile": {"ner_score": 0.9959202},
        }
        expected_swedish_locations = {
            "Uddevalla": {
                "place_id": 127650,
                "licence": "Data © OpenStreetMap contributors, ODbL 1.0. https://osm.org/copyright",
                "osm_type": "node",
                "osm_id": 26585398,
                "boundingbox": ["58.3090555", "58.3890555", "11.8982855", "11.9782855"],
                "lat": "58.3490555",
                "lon": "11.9382855",
                "display_name": "Uddevalla, Uddevalla kommun, Västra Götaland County, 451 30, Sweden",
                "class": "place",
                "type": "town",
                "importance": 0.6108750288615769,
                "icon": "https://nominatim.openstreetmap.org/ui/mapicons/poi_place_town.p.20.png",
                "extratags": {
                    "wikidata": "Q27447",
                    "wikipedia": "sv:Uddevalla",
                    "population": "31212",
                    "ref:se:scb": "4608",
                    "ref:se:pts:postort": "UDDEVALLA",
                },
            },
            "Ljungskile": {
                "place_id": 145672,
                "licence": "Data © OpenStreetMap contributors, ODbL 1.0. https://osm.org/copyright",
                "osm_type": "node",
                "osm_id": 29687437,
                "boundingbox": ["58.2046504", "58.2446504", "11.8998287", "11.9398287"],
                "lat": "58.2246504",
                "lon": "11.9198287",
                "display_name": "Ljungskile, Uddevalla kommun, Västra Götaland County, 459 30, Sweden",
                "class": "place",
                "type": "village",
                "importance": 0.48583894956154516,
                "icon": "https://nominatim.openstreetmap.org/ui/mapicons/poi_place_village.p.20.png",
                "extratags": {
                    "website": "http://www.ljungskile.se/",
                    "wikidata": "Q1721409",
                    "wikipedia": "sv:Ljungskile",
                    "ref:se:scb": "4492",
                    "ref:se:pts:postort": "LJUNGSKILE",
                },
            },
        }

        text = "Tyck till om länsstyrelsens riskhanteringsplan för översvämning https://t.co/3GlphvDKci #UAkommun #Uddevalla #Ljungskile"
        tokens = model.get_tokens(text)

        tokens_keys = ["entity", "index", "word", "start", "end"]
        assert all(
            [
                [token[key] == expected_token[key] for key in tokens_keys]
                for token, expected_token in zip(tokens, expected_tokens)
            ]
        )

        locations_tokens = model.get_location_tokens(tokens)
        assert locations_tokens.keys() == expected_locations_tokens.keys()

        swedish_locations = geocoder.get_swedish_location(locations_tokens)
        assert swedish_locations == expected_swedish_locations
