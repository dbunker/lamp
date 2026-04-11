d2(o3).
d1(o5).
d0(o5).
d0(o4).
d5(o1).
d0(o2).

d4(V) :- d0(V).
d3(V) :- not d1(V), d0(V).
d2(V) :- d5(V).
d5(V) :- d2(V).