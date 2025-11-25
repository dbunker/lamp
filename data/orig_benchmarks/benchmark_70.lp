d2(o4).
d3(o4).
d3(o3).
d1(o3).
d0(o4).

d1(X) :- d2(X), d0(X), not d3(X).
d1(X) :- d2(X), d0(X), not d3(X).
d0(X) :- not d2(X), d1(X), d3(X).