def get_centroid(polygon):
    centroid_str = polygon.centroid.wkt
    return shapely.wkt.loads(centroid_str)

def point_inside_shape(point, polygon) -> bool:
    point = gpd.GeoDataFrame(geometry=[point])
    return(point.within(polygon).iloc[0])

def rotated_square(cx, cy, size=70, degrees=0):
    """ Calculate coordinates of a rotated square or normal one centered at 'cx, cy'
        given its 'size' and rotation by 'degrees' about its center.
    """
    h = size/2
    l, r, b, t = cx-h, cx+h, cy-h, cy+h
    a = radians(degrees)
    cosa, sina = cos(a), sin(a)
    pts = [(l, b), (l, t), (r, t), (r, b)]
    return [(( (x-cx)*cosa + (y-cy)*sina) + cx,
             (-(x-cx)*sina + (y-cy)*cosa) + cy) for x, y in pts]

def make_shapely_points(points):
    return [Point(point) for point in points]

# other way around
def get_inner_rectangle_corner_coordinates_from_polygon(step_size,polygon):
    result_points = []
    rectangle_size = 70 # has enough buffer and suitable for all crowns
    centroid = get_centroid(polygon)
    rectangle_coordinates = rotated_square(centroid.x,centroid.y,rectangle_size,degrees=0) # initial square size
    while(rectangle_size != 0):
        result_points = rectangle_coordinates
        if all(point_inside_shape(Point(point),one_poly) for point in rectangle_coordinates):
            return result_points
        else: 
            # rectangle is outside of polygon
            rectangle_coordinates = rotated_square(centroid_geom.x,centroid_geom.y,rectangle_size,degrees=0)
            rectangle_size -= step_size