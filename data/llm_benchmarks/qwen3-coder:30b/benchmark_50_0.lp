d2(o1).
d1(o4).
d3(o3).
d1(o3).

d0(o3) :- not d1(o3), d3(o3).
d1(o1) :- not d0(o1), d2(o1).