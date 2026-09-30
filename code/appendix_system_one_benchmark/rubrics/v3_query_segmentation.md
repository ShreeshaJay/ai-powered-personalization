# Commerce-Query Segmentation Rubric v3

This is the frozen post-pilot rubric. It retains the v2 specificity hierarchy
and adds operational boundaries for the goal and object axes.

## Goal

- `known_item_navigation`: Reach a named model, title, person/author, distinct
  named item, brand/store destination, retailer, or website. A brand plus a
  generic product type is not automatically navigation.
- `product_discovery`: Find products satisfying a generic type, attributes,
  recipient, occasion, or broad need.
- `comparison_decision`: Compare alternatives, seek recommendations, or decide
  what to buy.
- `informational_support`: Learn, troubleshoot, obtain instructions, or answer
  a product-related question.
- `service_account`: Reach an account, login, order, payment, delivery,
  subscription, repair, or customer-service workflow.
- `other_unclear`: Malformed, non-commerce, or insufficiently clear.

## Object

Apply these boundaries independently of goal:

- `product`: A distinct named item, model, title, or identifier. Also use for a
  requested accessory when a named model constrains its fit.
- `category`: A generic product type or merchandise family, with or without
  ordinary constraints such as brand, color, size, material, or audience.
- `brand_store`: A brand, retailer, or storefront is the destination and no
  particular product type is requested.
- `service_content`: An account, service workflow, website/content destination,
  instructions, support, restaurant, or other non-product target.
- `unclear`: No defensible target object.

Examples:

- “Nike running shoes women” → category.
- “Pixel 3 screen protector” → product, because a named model constrains fit.
- “Montblanc” → brand_store.
- “Wells Fargo login” → service_content.

## Specificity

Apply the first matching level:

1. `exact_model_or_identifier`: Exact model, generation, part number, SKU-like
   identifier, or named model constraining an accessory.
2. `named_entity_or_title`: Brand/store only, person/author, media title,
   proprietary line, or named item without an exact model identifier.
3. `product_type_with_constraints`: Product type plus brand, size, color,
   material, audience, compatibility target without an exact model, quantity,
   feature, or other explicit constraint.
4. `product_type_only`: Generic product type/category with no explicit
   constraint.
5. `broad_need_or_occasion`: Broad need, activity, recipient, problem, theme,
   or occasion without one definite product type.
6. `none_or_unclear`: No defensible specificity.

Do not infer an exact identifier from an unexplained number.

## Commerce scope

- `in_scope`
- `out_of_scope`
- `ambiguous`

The judges work independently, use low effort, receive no weak labels, and
return structured JSON only.
