#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o4).
#constant(obj, o6).

#modeh(d3(var(obj))).
#modeh(d5(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d3(o0), d5(o0)
}, {
  d3(o2), d3(o4), d5(o2), d5(o4), d5(o6)
}, {
  d1(o0).
  d1(o2).
  d2(o0).
  d2(o4).
  d3(o6).
}).