d1(o4).
d1(o0).
d3(o3).
d1(o3).

d0(X) :- not d2(X), d1(X).
d0(X) :- d3(X).