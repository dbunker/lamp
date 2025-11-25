d1(o4).
d3(o4).
d3(o0).
d1(o3).
d0(o4).

d1(X) :- not d0(X), d2(X), not d3(X).
d2(X) :- not d1(X), not d0(X), d3(X).