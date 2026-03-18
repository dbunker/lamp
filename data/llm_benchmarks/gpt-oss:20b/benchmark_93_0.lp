d3(o1).
d4(o1).
d4(o2).
d1(o4).
d5(o2).
d2(o3).
d3(o4).
d4(o3).
d0(o4).

d3(X) :- not d2(X), d5(X).
d0(X) :- d5(X).