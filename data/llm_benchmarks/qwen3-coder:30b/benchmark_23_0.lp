d0(o3).
d3(o0).
d3(o1).
d1(o3).
d2(o0).

d2(X) :- d0(X).
d3(X) :- d2(X).
d2(X) :- not d0(X), d1(X).