d0(o3).
d1(o4).
d0(o2).
d3(o3).

d3(X) :- not d2(X), d1(X), d0(X).
d2(X) :- not d1(X), d0(X), d3(X).